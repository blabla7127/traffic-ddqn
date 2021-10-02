# 구현에 사용할 패키지 임포트
import numpy as np
#from numpy.lib.npyio import save
import torch
import gym
import time
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import os

FILENAME = int(time.time())
os.mkdir('anims/{}'.format(FILENAME))

# namedtuple 생성
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# 상수 정의
ENV = 'gym_traffic_sim:traffic-sim-v0'  # 태스크 이름
GAMMA = 0.99  # 시간할인율
MAX_STEPS = 500  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 50000  # 최대 에피소드 수

# transition을 저장하기 위한 메모리 클래스

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
import torch.multiprocessing as mp

import numpy as np

class ActorCriticNet(nn.Module):
    def __init__(self, n_in=(5+4*3*2), n_mid=32, n_out=32):
        super(ActorCriticNet, self).__init__()

        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

        self.Pi = nn.Linear(n_out, 2)
        self.V = nn.Linear(n_out, 1)

        self.initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        prob = self.Pi(x)
        prob = F.softmax(prob, dim=-1)

        value = self.V(x)

        return prob, value
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

# settings
learning_rate          = 1e-4
gamma                  = 0.99
beta                   = 0.01
max_T                  = 5000
max_t                  = 32
N_worker               = 8
model_path             = './Model_save.model'

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


############################################

def local_train(process, global_model, optimizer):
    env = gym.make(ENV)
    local_model = ActorCriticNet()
    local_model.load_state_dict(global_model.state_dict())

    total_reward =0
    max_score = 0

    for T in range(max_T):
        state_0, state_1 = env.reset()
        done_0 = False; done_1 = False
        score_0 = 0; score_1 = 0

        while not done_0:
            log_probs_0, values_0, entropys_0, rewards_0= [], [], [], []
            log_probs_1, values_1, entropys_1, rewards_1= [], [], [], []
            for t in range(max_t):
                prob_0, value_0 = local_model(torch.FloatTensor([[state_0]]))
                prob_1, value_1 = local_model(torch.FloatTensor([[state_1]]))
                
                m_0 = Categorical(prob_0)
                m_1 = Categorical(prob_1)

                action_0 = m_0.sample()
                action_1 = m_1.sample()
                log_prob_0 = m_0.log_prob(action_0)
                log_prob_1 = m_1.log_prob(action_1)
                entropy_0 = m_0.entropy()
                entropy_1 = m_1.entropy()

                (next_state_0, next_state_1), (reward_0, reward_1), (done_0, done_1), _ = env.step((action_0.item(), action_1.item()))
                
                score_0 += reward_0
                score_1 += reward_1

                log_probs_0.append(log_prob_0)
                log_probs_1.append(log_prob_1)
                values_0.append(value_0)
                values_1.append(value_1)
                entropys_0.append(entropy_0)
                entropys_1.append(entropy_1)
                rewards_0.append(reward_0)
                rewards_1.append(reward_1)

                state_0 = next_state_0
                state_1 = next_state_1
                if done_0:
                    break
            
            state_final_0 = torch.FloatTensor([next_state_0])
            state_final_1 = torch.FloatTensor([next_state_1])

            R_0 = 0.0
            if not done_0:
                _, R_0 = local_model(state_final_0)
                R_0 = R_0.item()
            R_1 = 0.0
            if not done_1:
                _, R_1 = local_model(state_final_1)
                R_1 = R_1.item()

            td_target_lst_0 = []
            td_target_lst_1 = []

            for reward_0 in rewards_0[::-1]:
                R_0 = reward_0 + R_0 * gamma
                td_target_lst_0.append([R_0])
            td_target_lst_0.reverse()

            for reward_1 in rewards_1[::-1]:
                R_1 = reward_1 + R_1 * gamma
                td_target_lst_1.append([R_1])
            td_target_lst_1.reverse()

            log_probs_0 = torch.stack(log_probs_0)
            log_probs_1 = torch.stack(log_probs_1)
            values_0 = torch.cat(values_0)
            values_1 = torch.cat(values_1)
            entropys_0 = torch.stack(entropys_0)
            entropys_1 = torch.stack(entropys_1)
            td_targets_0 = torch.FloatTensor(td_target_lst_0)
            td_targets_1 = torch.FloatTensor(td_target_lst_1)
            advantages_0 = (td_targets_0 - values_0).detach()
            advantages_1 = (td_targets_1 - values_1).detach()

            actor_loss_0 = -torch.mean(log_probs_0 * advantages_0)
            actor_loss_1 = -torch.mean(log_probs_1 * advantages_1)
            critic_loss_0 = F.smooth_l1_loss(values_0, td_targets_0.detach())
            critic_loss_1 = F.smooth_l1_loss(values_1, td_targets_1.detach())
            entropy_loss_0 = torch.mean(entropys_0)
            entropy_loss_1 = torch.mean(entropys_1)
            
            total_loss_0 = actor_loss_0 + critic_loss_0 - beta * entropy_loss_0
            total_loss_1 = actor_loss_1 + critic_loss_1 - beta * entropy_loss_1
            
            optimizer.zero_grad()
            local_model.zero_grad()

            total_loss_0.backward()
            total_loss_1.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5)

            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad
            
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())         

        score = score_0 + score_1
        total_reward += score
        if score > max_score:
            max_score = score

        if (T+1) % 10 == 0 :
            print('Process {} of episode {}, avg score : {}, max score : {}'.format(process, T+1, total_reward/10, max_score))
            total_reward = 0
    
    env.close()

def main():
    global_model = ActorCriticNet()
    #global_model.load_state_dict(torch.load(model_path))
    global_model.share_memory()

    optimizer = GlobalAdam(global_model.parameters(), learning_rate)
    
    processes = []

    for process in range(N_worker):
        p = mp.Process(target=local_train, args=(process, global_model, optimizer,))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    
    torch.save(global_model.state_dict(), model_path)

if __name__ == "__main__":
    main()