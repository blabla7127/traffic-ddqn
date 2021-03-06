# 구현에 사용할 패키지 임포트
import numpy as np
import torch
import time
FILENAME = int(time.time())
# namedtuple 생성
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# 상수 정의
ENV = 'CartPole-v0'  # 태스크 이름
GAMMA = 0.99  # 시간할인율
MAX_STEPS = 2000  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 100000  # 최대 에피소드 수

# transition을 저장하기 위한 메모리 클래스

device = torch.device("cuda")
#device = torch.device("cpu")
print(device)

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 메모리의 최대 저장 건수
        self.memory = []  # 실제 transition을 저장할 변수
        self.index = 0  # 저장 위치를 가리킬 인덱스 변수

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)을 메모리에 저장'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 메모리가 가득차지 않은 경우

        # Transition이라는 namedtuple을 사용하여 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 다음 저장할 위치를 한 자리 뒤로 수정

    def sample(self, batch_size):
        '''batch_size 갯수 만큼 무작위로 저장된 transition을 추출'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''len 함수로 현재 저장된 transition 갯수를 반환'''
        return len(self.memory)

# 신경망 구성
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = self.fc4(h3)
        return output

# 에이전트의 두뇌 역할을 하는 클래스, DDQN을 실제 수행한다 

import random
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPoleの行動（右に左に押す）の2を取得
        self.epsilon = 1
        # transition을 기억하기 위한 메모리 객체 생성
        self.memory = ReplayMemory(CAPACITY)

        # 신경망 구성
        n_in, n_mid, n_out = num_states, 64, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out).to(device)  # Net 클래스를 사용
        self.target_q_network = Net(n_in, n_mid, n_out).to(device)  # Net 클래스를 사용
        print(self.main_q_network)  # 신경망의 구조를 출력

        # 최적화 기법 선택
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replay로 신경망의 결합 가중치 학습'''

        # 1. 저장된 transition의 수를 확인
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 미니배치 생성
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 결합 가중치 수정
        self.update_main_q_network()

    def decide_action(self, state):
        '''현재 상태로부터 행동을 결정함'''
        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        #decay = 0.99995
        initial_epsilon = 1
        final_epsilon = 0.07
        
        if self.epsilon > final_epsilon:
            self.epsilon = (initial_epsilon - final_epsilon) / 7e6
        if self.epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # 신경망을 추론 모드로 전환
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            # .view(1,1)은 [torch.LongTensor of size 1] 을 size 1*1로 변환하는 역할을 한다

        else:
            # 행동을 무작위로 반환(0 혹은 1)
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(device)  # 행동을 무작위로 반환(0 혹은 1)
            # action은 [torch.LongTensor of size 1*1] 형태가 된다

        return action

    def make_minibatch(self):
        '''2. 미니배치 생성'''

        # 2.1 메모리 객체에서 미니배치를 추출
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이것을 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환한다
        batch = Transition(*zip(*transitions))

        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있도록 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 갯수만큼 있는 형태이다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates(연접)을 의미한다
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''정답신호로 사용할 Q(s_t, a_t)를 계산'''

        # 3.1 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태이다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 
        # 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather 메서드로 모아온다
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}값을 계산한다 이때 다음 상태가 존재하는지에 주의해야 한다

        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE).to(device)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor).to(device)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 결합 가중치 수정'''

        # 4.1 신경망을 학습 모드로 전환
        self.main_q_network.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은
        # size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()  # 경사를 초기화
        loss.backward()  # 역전파 계산
        self.optimizer.step()  # 결합 가중치 수정

    def update_target_q_network(self):  # DDQN에서 추가됨
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

# CartPole 태스크의 에이전트 클래스. 봉 달린 수레 자체라고 보면 된다


class Agent:
    def __init__(self, num_states, num_actions):
        '''태스크의 상태 및 행동의 가짓수를 설정'''
        self.brain = Brain(num_states, num_actions)  # 에이전트의 행동을 결정할 두뇌 역할 객체를 생성

    def update_q_function(self):
        '''Q함수를 수정'''
        self.brain.replay()

    def get_action(self, state):
        '''행동을 결정'''
        action = self.brain.decide_action(state)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memory 객체에 state, action, state_next, reward 내용을 저장'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.brain.update_target_q_network()
        
# CartPole을 실행하는 환경 클래스

from environment3 import Environment

class Environment_:

    def __init__(self):
        self.env = Environment()
        num_states = 4 * 3 * 2
        num_actions = 4
        self.agent = Agent(num_states, num_actions)  # 에이전트 역할을 할 객체를 생성

    def run(self):
        '''실행'''
        episode_10_list = np.zeros(10)  # 최근 10에피소드 동안 버틴 단계 수를 저장함(평균 단계 수를 출력할 때 사용)
        complete_episodes = 0  # 현재까지 195단계를 버틴 에피소드 수
        episode_final = False  # 마지막 에피소드 여부
        #frames = []  # 애니메이션을 만들기 위해 마지막 에피소드의 프레임을 저장할 배열
        observation_0 = torch.zeros(0).to(device)
        observation_1 = torch.zeros(0).to(device)
        state_0 = torch.zeros(0).to(device)
        state_1 = torch.zeros(0).to(device)

        iter = 0
        text = ('episodes:steps:mean_steps(10):rwd:mean_rwd:rwd_sum:rwd_sum-?*step:iter')
        log_ = open('{}.log'.format(FILENAME), 'a')
        log_.write(text+'\n')
        log_.close()

        for episode in range(NUM_EPISODES):  # 최대 에피소드 수만큼 반복
            obs, _, _ = self.env.reset()  # 환경 초기화
            observation_0, observation_1 = obs
            observation_0 = torch.from_numpy(np.array(observation_0)).float().to(device)
            observation_1 = torch.from_numpy(np.array(observation_1)).float().to(device)
            
            state_0 = observation_0.to(device)
            state_1 = observation_1.to(device)  # 관측을 변환없이 그대로 상태 s로 사용
            rwd = torch.zeros(1).to(device)

            # state_0 = torch.from_numpy(state_0).type(
            #     torch.FloatTensor)  # NumPy 변수를 파이토치 텐서로 변환
            # state_1 = torch.from_numpy(state_1).type(
            #     torch.FloatTensor)  # NumPy 변수를 파이토치 텐서로 변환
            
            state_0 = torch.unsqueeze(state_0, 0)  # size 4를 size 1*4로 변환
            state_1 = torch.unsqueeze(state_1, 0)  # size 4를 size 1*4로 변환

            for step in range(MAX_STEPS):  # 1 에피소드에 해당하는 반복문
                iter += 1
                # 애니메이션 만드는 부분을 주석처리
                #if episode_final is True:  # 마지막 에피소드에서는 각 시각의 이미지를 frames에 저장한다
                    # frames.append(self.env.render(mode='rgb_array'))
                
                action_0 = self.agent.get_action(state_0)  # 다음 행동을 결정
                action_1 = self.agent.get_action(state_1)  # 다음 행동을 결정
                # action_0 = action_0.squeeze(1).cpu().numpy()
                # action_1 = action_1.squeeze(1).cpu().numpy()

                # 행동 a_t를 실행하여 다음 상태 s_{t+1}과 done 플래그 값을 결정
                # action에 .item()을 호출하여 행동 내용을 구함
                (observation_next_0, observation_next_1), (rwd_0, rwd_1), (done_0, done_1) = self.env.step((action_0.item(), action_1.item()))  # reward와 info는 사용하지 않으므로 _로 처리
                observation_next_0 = np.array(observation_next_0)
                observation_next_1 = np.array(observation_next_1)
                if done_0:
                    if step < MAX_STEPS - 10:
                        rwd_0 = [-0.05]
                        rwd_1 = [-0.05]
                rwd_0 = torch.FloatTensor(rwd_0).to(device)
                rwd_1 = torch.FloatTensor(rwd_1).to(device)
                rwd += rwd_0 + rwd_1
                
#                if iter % 1000 == 0:
#                    print(iter, action_0.cpu().numpy(), action_1.cpu().numpy(), (rwd_0[0]+rwd_1[0]).cpu().numpy())
                    
                # 보상을 부여하고 episode의 종료 판정 및 state_next를 설정한다
                if done_0:  # 단계 수가 200을 넘었거나 봉이 일정 각도 이상 기울면 done이 True가 됨
                    state_next_0 = None
                    state_next_1 = None  # 다음 상태가 없으므로 None으로

                    # 최근 10 에피소드에서 버틴 단계 수를 리스트에 저장
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    if step < 195000:
                        complete_episodes = 0  # 연속 성공 에피소드 기록을 초기화
                    else:
                        complete_episodes = complete_episodes + 1  # 연속 성공 에피소드 기록을 갱신
                else:
                    state_next_0 = observation_next_0
                    state_next_1 = observation_next_1  # 관측 결과를 그대로 상태로 사용
                    state_next_0 = torch.from_numpy(state_next_0).type(
                        torch.FloatTensor).to(device)  # numpy 변수를 파이토치 텐서로 변환
                    state_next_1 = torch.from_numpy(state_next_1).type(
                        torch.FloatTensor).to(device)  # numpy 변수를 파이토치 텐서로 변환
                    state_next_0 = torch.unsqueeze(state_next_0, 0)  # size 4를 size 1*4로 변환
                    state_next_1 = torch.unsqueeze(state_next_1, 0)  # size 4를 size 1*4로 변환

                # 메모리에 경험을 저장
                self.agent.memorize(state_0, action_0, state_next_0, rwd_0)
                self.agent.memorize(state_1, action_1, state_next_1, rwd_1)

                # Experience Replay로 Q함수를 수정
                self.agent.update_q_function()
                if iter%500 == 0:
                    self.agent.update_target_q_function()
                # 관측 결과를 업데이트
                state_0 = state_next_0
                state_1 = state_next_1

                # 에피소드 종료 처리
                if done_0:
                    text = ('{0:7d}:{1:6d}:{2:9.1f}:{3:9.5f}:{4:9.5f}:{5:9.1f}:{6:10.5f}:{7:10d}').format(episode, step + 1, episode_10_list.mean(), (rwd_0[0]+rwd_1[0]).cpu().numpy(), rwd[0].cpu().numpy()/(step + 1), rwd[0].cpu().numpy(), rwd[0].cpu().numpy() - 0.1*(step + 1), iter)
                    print(text)
                    log_ = open('{}.log'.format(FILENAME), 'a')
                    log_.write(text+'\n')
                    log_.close()
                    break
                    
                    
            if episode_final is True:
                # 애니메이션 생성 부분을 주석처리함
                # 애니메이션 생성 및 저장
                #display_frames_as_gif(frames)
                break

            # 10 에피소드 연속으로 195단계를 버티면 태스크 성공
            if complete_episodes >= 10:
                pass
              #  print('10 에피소드 연속 성공')
              #  episode_final = True  # 다음 에피소드에서 애니메이션을 생성

# 실행 엔트리 포인트
cartpole_env = Environment_()
cartpole_env.run()
