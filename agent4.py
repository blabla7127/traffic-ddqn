import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import os

FILENAME = int(time.time())
os.mkdir('anims/{}'.format(FILENAME))

from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'gym_traffic_sim:traffic-sim-v0'
GAMMA = 0.99
MAX_STEPS = 500
NUM_EPISODES = 50000

########
LOAD_MODEL = True
########


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)을 메모리에 저장'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        '''batch_size 갯수 만큼 무작위로 저장된 transition을 추출'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''len 함수로 현재 저장된 transition 갯수를 반환'''
        return len(self.memory)


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        #self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)
        #self.fc4 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        #h3 = F.relu(self.fc3(h2))
        #output = self.fc4(h3)
        output = self.fc3(h2)
        return output


import random
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        if LOAD_MODEL == True:
            self.epsilon = 0.07
        else:
            self.epsilon = 1

        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out).to(device)
        self.target_q_network = Net(n_in, n_mid, n_out).to(device)
        print(self.main_q_network)

        self.optimizer = optim.RMSprop(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state):
        initial_epsilon = 1
        middle_epsilon = 0.25
        final_epsilon = 0.01

        decay = np.power(middle_epsilon/initial_epsilon, 1/1e6)
        decay2 = np.power(final_epsilon/middle_epsilon, 1/6e6)

        decay = np.sqrt(decay)
        decay2 = np.sqrt(decay2)
        
        if self.epsilon > middle_epsilon:
            self.epsilon *= decay
        elif self.epsilon > final_epsilon:
            self.epsilon *= decay2

        if self.epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(device)

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))

        next_state_values = torch.zeros(BATCH_SIZE).to(device)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor).to(device)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state):
        action = self.brain.decide_action(state)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

import gym

class Environment_:

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = 5 + 4 * 3 * 2
        num_actions = 2
        self.agent = Agent(num_states, num_actions)
        if LOAD_MODEL == True:
            self.agent.brain.main_q_network.load_state_dict(torch.load('./Model_save.model'))
            self.agent.brain.target_q_network.load_state_dict(torch.load('./Model_save.model'))

    def run(self):
        complete_episodes = 0

        observation_0 = torch.zeros(0).to(device)
        observation_1 = torch.zeros(0).to(device)
        state_0 = torch.zeros(0).to(device)
        state_1 = torch.zeros(0).to(device)

        iter = 0
        text = ('episodes:steps:(deprecated):rwd:mean_rwd:rwd_sum:epsilon:iter')
        log_ = open('logs/{}.log'.format(FILENAME), 'a')
        log_.write(text+'\n')
        log_.close()

        for episode in range(NUM_EPISODES):
            obs = self.env.reset()
            observation_0, observation_1 = obs
            observation_0 = torch.from_numpy(np.array(observation_0)).float().to(device)
            observation_1 = torch.from_numpy(np.array(observation_1)).float().to(device)
            score = 0

            state_0 = observation_0.to(device)
            state_1 = observation_1.to(device)
            rwd = torch.zeros(1).to(device)

            state_0 = torch.unsqueeze(state_0, 0)
            state_1 = torch.unsqueeze(state_1, 0)

            save_anim = False
            if episode % 1000 == 0:
                save_anim = True
                frames = []

            for step in range(MAX_STEPS):
                iter += 1
                
                action_0 = self.agent.get_action(state_0)
                action_1 = self.agent.get_action(state_1)

                (observation_next_0, observation_next_1), (rwd_0, rwd_1), (done_0, done_1), (score_0, score_1) = self.env.step((action_0.item(), action_1.item()))
                if save_anim:
                    frames.append((self.env.render(), rwd_0, rwd_1))
                observation_next_0 = np.array(observation_next_0)
                observation_next_1 = np.array(observation_next_1)

                rwd_0 = torch.FloatTensor(rwd_0).to(device)
                rwd_1 = torch.FloatTensor(rwd_1).to(device)
                rwd += rwd_0 + rwd_1

                score += score_0[0] + score_1[0]

                if done_0:
                    state_next_0 = None
                    state_next_1 = None

                else:
                    state_next_0 = observation_next_0
                    state_next_1 = observation_next_1
                    state_next_0 = torch.from_numpy(state_next_0).type(
                        torch.FloatTensor).to(device)
                    state_next_1 = torch.from_numpy(state_next_1).type(
                        torch.FloatTensor).to(device)
                    state_next_0 = torch.unsqueeze(state_next_0, 0)
                    state_next_1 = torch.unsqueeze(state_next_1, 0)

                self.agent.memorize(state_0, action_0, state_next_0, rwd_0)
                self.agent.memorize(state_1, action_1, state_next_1, rwd_1)

                self.agent.update_q_function()

                if iter%1000 == 0:
                    self.agent.update_target_q_function()

                state_0 = state_next_0
                state_1 = state_next_1

                if done_0:
                    text = ('{0:7d}:{1:6d}:{2:9.1f}:{3:9.5f}:{4:9.5f}:{5:9.1f}:{6:9.7f}:{7:10d}').format(episode, step + 1, 0, (rwd_0[0]+rwd_1[0]).cpu().numpy(), rwd[0].cpu().numpy()/(step + 1), rwd[0].cpu().numpy(), self.agent.brain.epsilon, iter)
                    print(text)
                    log_ = open('logs/{}.log'.format(FILENAME), 'a')
                    log_.write(text+'\n')
                    log_.close()
                    self.env.reset()
                    break
            
            if save_anim:
                torch.save(self.agent.brain.main_q_network.state_dict(), './Model_save.model')
                fig, ax = plt.subplots(1,1)
                artists = []

                for frame, rwd0, rwd1 in frames:
                    ms = ax.matshow(frame)
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    title = plt.text(0.5,1.01,'{0:8.5f},        {1:8.5f}'.format(rwd0[0], rwd1[0]), ha="center",va="bottom",
                                transform=ax.transAxes, fontsize="large")
                    artists.append([ms,title])
                ani = ArtistAnimation(fig, artists, interval=100)
                ani.save('anims/{}/{}.gif'.format(FILENAME,episode), dpi = 200)        

# 실행 엔트리 포인트
cartpole_env = Environment_()
cartpole_env.run()
