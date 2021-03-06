import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

#LEN_ROAD = 200
class TrafficSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.iter = 0
        self.ang_0 = 0
        self.ang_1 = 0
        self.LEN_ROAD = 50
        #self.random_table = np.random.random_sample((1000)) < 0.9
        #self.random_table_pos = 0
        self.route = np.array([
                        [[ 0, 0],[ 5, 1]],
                        [[ 0, 2],[-1,-1]],
                        [[ 0, 1],[-1,-1]],
                        [[ 0, 0],[ 5, 2]],
                        [[ 1, 2],[ 2, 1]],
                        [[ 1, 0],[-1,-1]],
                        [[ 1, 2],[ 2, 0]],
                        [[ 1, 1],[-1,-1]],
                        [[ 4, 1],[ 5, 1]],
                        [[ 4, 2],[-1,-1]],
                        [[ 4, 1],[ 5, 2]],
                        [[ 3, 1],[ 2, 0]],
                        [[ 3, 0],[-1,-1]],
                        [[ 6, 2],[ 5, 2]],
                        [[ 7, 0],[ 2, 0]],
                        [[ 7, 2],[-1,-1]],
                        [[ 7, 0],[ 2, 1]],
                        [[ 7, 1],[-1,-1]],
                        [[ 7, 0],[ 2, 2]],
                        [[ 6, 2],[ 5, 1]],
                        [[ 6, 0],[-1,-1]],
                        [[ 6, 2],[ 5, 0]],
                        [[ 6, 1],[-1,-1]],
                        [[ 3, 1],[ 2, 1]],
                        [[ 3, 2],[-1,-1]],
                        [[ 3, 1],[ 2, 2]],
                        [[ 4, 1],[ 5, 0]],
                        [[ 4, 0],[-1,-1]],
                        [[ 1, 2],[ 2, 2]],
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
        self.cars_buf = np.zeros((self.len_cars_buf,3,2), np.int8)
        self.carbuf_pointer = 0
        self.linebuf_pointer = np.zeros((8,3), np.int8)
        self.done = False
        self.trafficlight = np.zeros((8,3), np.bool8)

        return ((*self.queue_length_per_line[[0,4,6,2]].flatten(), *self.mean_waittime_per_line[[0,4,6,2]].flatten()),(*self.queue_length_per_line[[1,5,7,3]].flatten(), *self.mean_waittime_per_line[[1,5,7,3]].flatten()))

    def adjust_trafficlight(self, action):
        a0, a1 = action
        self.trafficlight[[0,2,4,6]] = self.trafficlight_types[a0]
        self.trafficlight[[1,3,5,7]] = self.trafficlight_types[a1]

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
        self.cars_buf[self.carbuf_pointer, :-1] = self.route[np.random.randint(0,29)]
        self.cars_buf[self.carbuf_pointer, -1] = (0,-1)
        line_0_0, line_0_1 = self.cars_buf[self.carbuf_pointer, 0]
        
        asdf = self.add_line_buf(line_0_0, line_0_1, self.carbuf_pointer)
        if asdf == 1:
            return

        while self.cars_buf[self.carbuf_pointer, -1, 1] == -1:
            self.carbuf_pointer = (self.carbuf_pointer + 1) % self.len_cars_buf

    def move_car(self):
        mask_is_valid = self.lines_buf[...,5] != 0
        pos = self.lines_buf[...,2]
        speed = self.lines_buf[...,4]
        mask_is_in_queue = (self.lines_buf[..., 2] < self.LEN_ROAD//2) & (self.lines_buf[..., 2] >= 0) & (mask_is_valid)
        __index_of_front = self.lines_buf[..., 0]
        for i in range(8):
            for j in range(3):
                queue_len = 0
                waittime_sum = 0

                head, tail = self.line_ends[i,j]
                
                if head != -1:
                    mask_is_valid_ = mask_is_valid[i,j]
                    pos_ = pos[i,j]
                    speed_ = speed[i,j]
                    pos_[mask_is_valid_] -= speed_[mask_is_valid_]

                    mask_is_in_queue_ = mask_is_in_queue[i,j]

                    queue_len = mask_is_in_queue_.sum()
                    self.lines_buf[i,j,mask_is_in_queue_, 6] += 1
                    waittime_sum = np.sum(self.lines_buf[i,j,mask_is_in_queue_, 6])
                    self.queue_length_per_line[i,j] = queue_len
                    self.mean_waittime_per_line[i,j] = waittime_sum / (queue_len + 1)


                    s_plus = speed_+1
                    speed_inc = 2*(2 < s_plus) + (s_plus)*(2 >= s_plus)
                    greenlight = self.trafficlight[i,j]

                    if greenlight or pos_[head] >= 3:
                        speed_[head] = speed_inc[head]
                    elif pos_[head] > 0:
                        speed_[head] = 1
                    else:
                        speed_[head] = 0

                    __index_of_front_ = __index_of_front[i,j]
                    front_pos = pos_[__index_of_front_]
                    distance = pos_ - front_pos
                    
                    mask_not_head = (self.lines_buf[i,j,head,3] != self.lines_buf[i,j,:,3])
                    mask_0 = (distance <= 2)
                    mask_1 = ((~mask_0) & (distance <= 6))
                    mask_2 = ~(mask_0 | mask_1)

                    speed_[(mask_0 & mask_not_head & mask_is_valid_)] = 0
                    speed_[(mask_1 & mask_not_head & mask_is_valid_)] = 1
                    speed_[(mask_2 & mask_not_head & mask_is_valid_)] = speed_inc[(mask_2 & mask_not_head & mask_is_valid_)]

##################################################################
        for i in range(8):
            for j in range(3):
                head, tail = self.line_ends[i,j]
                if head != -1 and self.lines_buf[i,j,head,2] < 0:

                    head_id = self.lines_buf[i,j,head,3]
                    cur_step = self.cars_buf[head_id, -1, 0]
                    
                    l0, l1 = self.cars_buf[head_id, cur_step + 1]
                    if l1 == -1:
                        self.rmv_line_buf(i,j)
                        self.cars_buf[head_id, -1, 0] = cur_step + 1
                        self.cars_buf[head_id, -1, 1] = 0
                    else:
                        asdf = self.add_line_buf(l0, l1, head_id)
                        if asdf == 0:
                            self.rmv_line_buf(i,j)
                            self.cars_buf[head_id, -1, 0] = cur_step + 1
                        else:
                            self.lines_buf[i,j,head,4] = 0

    def step(self, action : tuple):
        self.iter += 1

        self.adjust_trafficlight(action)
        for _ in range(2):
            self.create_car()
        self.move_car()

        rms_queue_len_0 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[0,4,6,2]])))
        rms_queue_len_1 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[1,5,7,3]])))
        rms_waittime_0 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[0,4,6,2]], dtype=np.int32)))
        rms_waittime_1 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[1,5,7,3]], dtype=np.int32)))
        aang_0 = rms_queue_len_0 * rms_waittime_0
        aang_1 = rms_queue_len_1 * rms_waittime_1

        rwd_0 = np.array((0.05 + (self.ang_0 - aang_0),))/10
        rwd_1 = np.array((0.05 + (self.ang_1 - aang_1),))/10

        self.ang_0 = aang_0
        self.ang_1 = aang_1

        if self.iter == 2000:
            self.done = True
        ret = (((*self.queue_length_per_line[[0,4,6,2]].flatten(), *self.mean_waittime_per_line[[0,4,6,2]].flatten()),(*(self.queue_length_per_line[[1,5,7,3]].flatten()), *self.mean_waittime_per_line[[1,5,7,3]].flatten())), (rwd_0, rwd_1), (self.done, self.done), None)

        return ret
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

def main():
    env = TrafficSimEnv()

    for i in range(200000):
        
        action1 = np.random.randint(0,4)
        action2 = np.random.randint(0,4)

        obs, rwd, done, _ = env.step((action1, action2))
        if env.iter%2000 == 0:
            print(obs)
            print(env.iter)
        if done[0]:
            print(obs)
            print(env.iter)
            print(rwd)
            print('---')
            env.reset()


if __name__ == '__main__':
    main()
    
