import gym
from gym import error, spaces, utils
from gym.core import ActionWrapper
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

#LEN_ROAD = 200
class TrafficSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.iter = 0
        self.ang_0 = 0
        self.ang_1 = 0
        self.LEN_ROAD = 50
        self.route = np.array([
                        [[ 0, 0],[ 5, 1],[-1,-1]],
                        [[ 0, 2],[-1,-1],[-1,-1]],
                        [[ 0, 1],[-1,-1],[-1,-1]],
                        [[ 0, 0],[ 5, 2],[-1,-1]],
                        [[ 1, 2],[ 2, 1],[-1,-1]],
                        [[ 1, 0],[-1,-1],[-1,-1]],
                        [[ 1, 2],[ 2, 0],[-1,-1]],
                        [[ 1, 1],[-1,-1],[-1,-1]],
                        [[ 4, 1],[ 5, 1],[-1,-1]],
                        [[ 4, 2],[-1,-1],[-1,-1]],
                        [[ 4, 1],[ 5, 2],[-1,-1]],
                        [[ 3, 1],[ 2, 0],[-1,-1]],
                        [[ 3, 0],[-1,-1],[-1,-1]],
                        [[ 6, 2],[ 5, 2],[-1,-1]],
                        [[ 7, 0],[ 2, 0],[-1,-1]],
                        [[ 7, 2],[-1,-1],[-1,-1]],
                        [[ 7, 0],[ 2, 1],[-1,-1]],
                        [[ 7, 1],[-1,-1],[-1,-1]],
                        [[ 7, 0],[ 2, 2],[-1,-1]],
                        [[ 6, 2],[ 5, 1],[-1,-1]],
                        [[ 6, 0],[-1,-1],[-1,-1]],
                        [[ 6, 2],[ 5, 0],[-1,-1]],
                        [[ 6, 1],[-1,-1],[-1,-1]],
                        [[ 3, 1],[ 2, 1],[-1,-1]],
                        [[ 3, 2],[-1,-1],[-1,-1]],
                        [[ 3, 1],[ 2, 2],[-1,-1]],
                        [[ 4, 1],[ 5, 0],[-1,-1]],
                        [[ 4, 0],[-1,-1],[-1,-1]],
                        [[ 1, 2],[ 2, 2],[-1,-1]],
                        ])
        self.trafficlight_types = np.array([
            [
                [0,1,1],
                [0,0,0],
                [0,0,0],
                [0,1,1]
            ],
            [
                [0,0,0],
                [0,1,1],
                [0,1,1],
                [0,0,0]
            ],
            [
                [0,0,0],
                [1,0,0],
                [1,0,0],
                [0,0,0]
            ],
            [
                [1,0,0],
                [0,0,0],
                [0,0,0],
                [1,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ]

        ], np.bool8)
        self.actions = ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3))
        self.len_lines_buf = (self.LEN_ROAD)//2 + 5
        self.len_cars_buf = self.len_lines_buf * 8 * 3
        self.reset()


    def reset(self):
        self.iter = 0
        self.ang_0 = 0
        self.ang_1 = 0
        self.lines_buf = np.zeros((8,3,self.len_lines_buf,7), np.int16)
        self.lines_buf[..., 3] -= 1
        self.mean_waittime_per_line = np.zeros((8,3), np.int16)
        self.queue_length_per_line = np.zeros((8,3), np.int16)
        self.line_ends = np.zeros((8,3,2), np.int8) # head, tail
        self.line_ends -= 1
        self.cars_buf = np.zeros((self.len_cars_buf,2), np.int8)
        self.cars_buf[...,1] -= 1
        self.carbuf_pointer = 0
        self.linebuf_pointer = np.zeros((8,3), np.int8)
        self.done = False
        self.trafficlight = np.zeros((8,3), np.bool8)
        self.traffic_status = np.zeros(2, dtype=np.int8) - 1
        self.traffic_pointer = np.zeros(2, dtype=np.int8) + 3

        self.exit_loop = 0

        return ((self.traffic_pointer[0], *self.queue_length_per_line[[0,4,6,2]].flatten(), *(self.mean_waittime_per_line[[0,4,6,2]] / 10).flatten()),(self.traffic_pointer[1], *self.queue_length_per_line[[1,5,7,3]].flatten(), *(self.mean_waittime_per_line[[1,5,7,3]] / 10).flatten()))

    def adjust_trafficlight(self, action):
        foo = [[0,2,4,6],[1,3,5,7]]
        for iter, iaction in zip([0,1],action):
            if iaction != self.traffic_status[iter]:
                self.traffic_pointer[iter] = (self.traffic_pointer[iter] + 1) % 4
                if self.traffic_pointer[iter] == 3:
                    self.trafficlight[foo[iter]] = self.trafficlight_types[iaction]
                    self.traffic_status[iter] = iaction
                else:
                    self.trafficlight[foo[iter]] = self.trafficlight_types[-1]
            else:
                self.traffic_pointer[iter] = 3

    def add_line_buf(self, line_0, line_1, car_id):
        linebuf_pointer = self.linebuf_pointer[line_0, line_1]
        head, tail = self.line_ends[line_0, line_1]
        tail_pos = self.lines_buf[line_0, line_1, tail, 2]
        tail_speed = self.lines_buf[line_0, line_1, tail, 4]

        if tail != -1 and tail_pos >= self.LEN_ROAD - 2 and tail_speed <= 1:
            return 1
        if tail != -1:
            self.lines_buf[line_0, line_1, tail, 1] = linebuf_pointer
        else:
            self.line_ends[line_0, line_1, 0] = linebuf_pointer
        self.line_ends[line_0, line_1, 1] = linebuf_pointer

        if tail_pos >= self.LEN_ROAD - 2 and tail_speed == 2:
            speed = 1 
        else:
            speed = 2
        self.lines_buf[line_0, line_1, linebuf_pointer] = (tail, -1, self.LEN_ROAD, car_id, speed, 1 if line_1 == 1 else -1, 0)
        self.linebuf_pointer[line_0, line_1] = (linebuf_pointer + 1) % self.len_lines_buf

        return 0


    def rmv_line_buf(self, line_0, line_1):
        head, tail = self.line_ends[line_0, line_1]
        self.lines_buf[line_0,line_1,head,5] = 0
        rear = self.lines_buf[line_0,line_1,head,1]
        self.lines_buf[line_0,line_1,rear,0] = -1

        if head == tail:
            tail = -1
            self.line_ends[line_0,line_1,1] = tail

        head = rear
        if head != -1:
            self.lines_buf[line_0,line_1,head,0] = -1
        self.line_ends[line_0,line_1,0] = head

    def create_car(self):
        self.cars_buf[self.carbuf_pointer, 0] = np.random.randint(0,29)
        line_0_0, line_0_1 = self.route[self.cars_buf[self.carbuf_pointer, 0], 0]
        #print(line_0_0, line_0_1, end=' ')
        if self.add_line_buf(line_0_0, line_0_1, self.carbuf_pointer) == 1:
            if self.exit_loop < 10:
                self.exit_loop += 1
                self.create_car()
            self.exit_loop = 0
            return

        #print(self.cars_buf[self.carbuf_pointer, 0])
        self.cars_buf[self.carbuf_pointer, 1] = 0

        while self.cars_buf[self.carbuf_pointer, 1] == 0:
            self.carbuf_pointer = (self.carbuf_pointer + 1) % self.len_cars_buf

    def move_car(self):
        mask_is_valid = self.lines_buf[...,5] != 0
        pos = self.lines_buf[...,2]
        speed = self.lines_buf[...,4]
        pos -= speed
        mask_is_in_queue = (self.lines_buf[..., 2] < (self.LEN_ROAD//2)) & (self.lines_buf[..., 2] >= 0) & (mask_is_valid)
        queue_len = mask_is_in_queue.sum(axis=-1)
        waittime_sum = np.zeros((8,3))
        for i in range(8):
            for j in range(3):
                self.lines_buf[i,j,mask_is_in_queue[i,j],6] += 1
                waittime_sum[i,j] = np.sum(self.lines_buf[i,j,mask_is_in_queue[i,j],6])
        #print(pos)
        #waittime_sum = np.array([[np.sum(self.lines_buf[i,j,mask_is_in_queue[i,j],6]) for j in range(3)] for i in range(8)])
        self.queue_length_per_line = queue_len
        self.mean_waittime_per_line = waittime_sum / (queue_len + 1)


        #s_plus = speed + 1
        #speed_inc = 2*(2 < s_plus) + (s_plus)*(2 >= s_plus)
        __index_of_front = self.lines_buf[..., 0]
        
        for i in range(8):
            for j in range(3):

                head, tail = self.line_ends[i,j]
                
                if head != -1:
                    mask_is_valid_ = mask_is_valid[i,j]
                    pos_ = pos[i,j]
                    speed_ = speed[i,j]
                    #speed_inc_ = speed_inc[i,j]

                    __index_of_front_ = __index_of_front[i,j]
                    front_pos = pos_[__index_of_front_]
                    distance = pos_ - front_pos
                    
                    mask_0 = (distance <= 2)
                    mask_1 = ((~mask_0) & (distance <=3))
                    mask_2 = ~(mask_0 | mask_1)

                    speed_[(mask_0 & mask_is_valid_)] = 0
                    speed_[(mask_1 & mask_is_valid_)] = 1
                    speed_[(mask_2 & mask_is_valid_)] = 2#speed_inc_[(mask_2 & mask_is_valid_)]

                    greenlight = self.trafficlight[i,j]
                    if greenlight or pos_[head] >= 3:
                        s_plus = speed_[head] + 1
                        speed_[head] = 2*(2 < s_plus) + (s_plus)*(2 >= s_plus)
                    elif pos_[head] > 0:
                        speed_[head] = 1
                    else:
                        speed_[head] = 0

        for i in range(8):
            for j in range(3):
                head, tail = self.line_ends[i,j]
                if head != -1 and self.lines_buf[i,j,head,2] < 0:

                    head_id = self.lines_buf[i,j,head,3]
                    cur_step = self.cars_buf[head_id, 1]
                    
                    l0, l1 = self.route[self.cars_buf[head_id, 0], cur_step + 1]
                    if l1 == -1:
                        self.rmv_line_buf(i,j)
                        self.cars_buf[head_id, 1] = -1
                    else:
                        asdf = self.add_line_buf(l0, l1, head_id)
                        if asdf == 0:
                            self.rmv_line_buf(i,j)
                            self.cars_buf[head_id, 1] = cur_step + 1
                        else:
                            #print(i,j,l0,l1, cur_step+1, self.cars_buf[head_id,0])
                            self.lines_buf[i,j,head,4] = 0

    def step(self, action : tuple):
        self.iter += 1

        self.adjust_trafficlight(action)
        self.create_car()
        self.move_car()

        # rms_queue_len_0 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[0,4,6,2]])))
        # rms_queue_len_1 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[1,5,7,3]])))
        # rms_waittime_0 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[0,4,6,2]])))
        # rms_waittime_1 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[1,5,7,3]])))
        # aang_0 = rms_queue_len_0 * rms_waittime_0
        # aang_1 = rms_queue_len_1 * rms_waittime_1

        aang_0 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[0,4,6,2]]*self.queue_length_per_line[[0,4,6,2]])))/1000
        aang_1 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[1,5,7,3]]*self.queue_length_per_line[[1,5,7,3]])))/1000
    
        score_0 = np.array(((-aang_0),))
        score_1 = np.array(((-aang_1),))
        rwd_0 = score_0 #(np.array(((self.ang_0 - aang_0),)) * 250 + score_0) / 2
        rwd_1 = score_1 #(np.array(((self.ang_1 - aang_1),)) * 250 + score_1) / 2
        
        self.ang_0 = aang_0
        self.ang_1 = aang_1

        if self.iter == 500:
            self.done = True
        ret = (((self.traffic_pointer[0], *self.queue_length_per_line[[0,4,6,2]].flatten(), *(self.mean_waittime_per_line[[0,4,6,2]] / 10).flatten()),(self.traffic_pointer[1], *(self.queue_length_per_line[[1,5,7,3]].flatten()), *(self.mean_waittime_per_line[[1,5,7,3]] / 10).flatten())), (rwd_0, rwd_1), (self.done, self.done), (score_0, score_1))

        return ret
    
    def render(self, mode='human'):
        nx = 165
        ny = 108

        frame = np.zeros((ny,nx))
        frame[0:51,0:51] = 1
        frame[0:51,57:108] = 1
        frame[0:51,114:165] = 1
        frame[57:108,0:51] = 1
        frame[57:108,57:108] = 1
        frame[57:108,114:165] = 1

        frame[51:57,51:57] = 2
        frame[51:57,108:114] = 2
        
        frame[51,51:54] = self.trafficlight[0,::-1] + 2
        frame[51,108:111] = self.trafficlight[1,::-1] + 2
        frame[51:54,56] = self.trafficlight[2,::-1].transpose() + 2
        frame[51:54,113] = self.trafficlight[3,::-1].transpose() + 2
        frame[54:57,51] = self.trafficlight[4].transpose() + 2
        frame[54:57,108] = self.trafficlight[5].transpose() + 2
        frame[56,54:57] = self.trafficlight[6] + 2
        frame[56,111:114] = self.trafficlight[7] + 2

        lines = np.zeros((8,3,51), dtype=np.int8)
        for i in range(8):
            for j in range(3):
                head, tail = self.line_ends[i,j]
                cur = head
                if head == -1:
                    continue
                while True:
                    pos = self.lines_buf[i,j,cur,2]
                    
                    if pos >= 0 and pos <= 50:
                        lines[i,j,pos] = 4
                    cur = self.lines_buf[i,j,cur,1]
                    if cur == -1:
                        break
        # print(lines[0].transpose())
        # print(frame[0:51:-1,51:54:-1])
        frame[50::-1,53:50:-1] = lines[0].transpose()
        frame[50::-1,110:107:-1] = lines[1].transpose()
        frame[53:50:-1,57:108] = lines[2]
        frame[53:50:-1,114:165] = lines[3]
        frame[54:57,50::-1] = lines[4]
        frame[54:57,107:56:-1] = lines[5]
        frame[57:108,54:57] = lines[6].transpose()
        frame[57:108,111:114] = lines[7].transpose()

        return frame

    def close(self):
        pass

from matplotlib.animation import ArtistAnimation
def main():
    env = TrafficSimEnv()
    frames = []
    rwd_sum = 0
    scr_sum = 0
    asdf = 0
    asd = 15
    for i in range(500):
        if asdf < asd:
            action = 0
        elif asdf < asd*2:
            action = 1
        elif asdf < asd*3:
            action = 2
        else:
            action = 3
        asdf = (asdf + 1) % (asd * 4)
        
        action1 = action
        action2 = action

        # action1 = np.random.randint(0,4)
        # action2 = np.random.randint(0,4)
        
        obs, rwd, done, scr = env.step((action1, action2))
        frames.append((env.render(),rwd[0],rwd[1]))
        rwd_sum += rwd[0] + rwd[1]
        scr_sum += scr[0] + scr[1]
        if env.iter%1000 == 0:
            print(obs)
            print(env.iter)
        if done[0]:
            print(obs)
            print(env.iter)
            print(rwd_sum[0])
            print(scr_sum[0])
            print('---')
            env.reset()

    fig, ax = plt.subplots()
    artists = []

    for frame, rwd0, rwd1 in frames:
        ms = ax.matshow(frame)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        title = plt.text(0.5,1.01,'{0:8.5f},        {1:8.5f}'.format(rwd0[0], rwd1[0]), ha="center",va="bottom",
                    transform=ax.transAxes, fontsize="large")
        artists.append([ms,title])
    ani = ArtistAnimation(fig, artists, interval=100)
    print('a')
    ani.save('{}.gif'.format(asd), dpi = 200)
    print('b')
    plt.show()


if __name__ == '__main__':
    main()
    
