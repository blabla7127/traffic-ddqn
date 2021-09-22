import numpy as np

#LEN_ROAD = 200
class Environment:
    def __init__(self):
        self.iter = 0
        self.ang_0 = 0
        self.ang_1 = 0
        self.LEN_ROAD = 50
        #self.random_table = np.random.random_sample((1000)) < 0.9
        #self.random_table_pos = 0
        self.route = np.array([
                        [[ 0, 0],[ 5, 1],[ 5, 0]],
                        [[ 0, 3],[-1,-1],[-1,-1]],
                        [[ 0, 0],[ 5, 2],[-1,-1]],
                        [[ 0, 2],[-1,-1],[-1,-1]],
                        [[ 0, 0],[ 5, 3],[-1,-1]],
                        [[ 1, 3],[ 2, 2],[-1,-1]],
                        [[ 1, 0],[-1,-1],[-1,-1]],
                        [[ 1, 3],[ 2, 0],[-1,-1]],
                        [[ 1, 2],[-1,-1],[-1,-1]],
                        [[ 4, 2],[ 5, 2],[-1,-1]],
                        [[ 4, 3],[-1,-1],[-1,-1]],
                        [[ 4, 2],[ 5, 3],[-1,-1]],
                        [[ 3, 1],[ 2, 0],[-1,-1]],
                        [[ 3, 0],[-1,-1],[-1,-1]],
                        [[ 6, 3],[ 5, 3],[-1,-1]],
                        [[ 7, 0],[ 2, 0],[-1,-1]],
                        [[ 7, 3],[-1,-1],[-1,-1]],
                        [[ 7, 0],[ 2, 2],[-1,-1]],
                        [[ 7, 2],[-1,-1],[-1,-1]],
                        [[ 7, 0],[ 2, 3],[-1,-1]],
                        [[ 6, 3],[ 5, 2],[-1,-1]],
                        [[ 6, 0],[-1,-1],[-1,-1]],
                        [[ 6, 3],[ 5, 0],[-1,-1]],
                        [[ 6, 2],[-1,-1],[-1,-1]],
                        [[ 3, 2],[ 2, 2],[-1,-1]],
                        [[ 3, 3],[-1,-1],[-1,-1]],
                        [[ 3, 2],[ 2, 3],[-1,-1]],
                        [[ 4, 1],[ 5, 0],[-1,-1]],
                        [[ 4, 0],[-1,-1],[-1,-1]],
                        [[ 1, 3],[ 2, 3],[-1,-1]],
                        ])

        self.actions = ((0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3))
        self.len_lines_buf = (self.LEN_ROAD)+4
        self.lines_buf = np.zeros((8,4,self.len_lines_buf,7), np.int16)
        self.lines_buf[..., 3] -= 1
        self.mean_waittime_per_line = np.zeros((8,4), np.int16)
        self.queue_length_per_line = np.zeros((8,4), np.int16)
        self.line_ends = np.zeros((8,4,2), np.int8) # head, tail
        self.line_ends -= 1
        self.len_cars_buf = (self.LEN_ROAD)*8
        self.cars_buf = np.zeros((self.len_cars_buf,4,2), np.int8)
        self.carbuf_pointer = 0
        self.linebuf_pointer = np.zeros((8,4), np.int8)
        self.fail_0 = 0
        self.fail_1 = 0
        self.done = False
        self.spawn_car = 0
        self.trafficlight = np.zeros((8,4), np.int8)

    def reset(self):
        self.__init__()
        return (((*self.queue_length_per_line[[0,4,6,2]].flatten(), *self.mean_waittime_per_line[[0,4,6,2]].flatten()),(*self.queue_length_per_line[[1,5,7,3]].flatten(), *self.mean_waittime_per_line[[1,5,7,3]].flatten())), (0,0), (0,0))

    def adjust_trafficlight(self, action):
        #####a1, a2 = self.actions[action]
        a0, a1 = action

        trafficlight_types = np.array([
            [
                [0,1,1,1],
                [0,0,0,0],
                [0,0,0,0],
                [0,1,1,1]
            ],
            [
                [0,0,0,0],
                [0,1,1,1],
                [0,1,1,1],
                [0,0,0,0]
            ],
            [
                [0,0,0,0],
                [1,0,0,0],
                [1,0,0,0],
                [0,0,0,0]
            ],
            [
                [1,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [1,0,0,0]
            ]

        ], np.bool8)

        self.trafficlight[[0,2,4,6]] = trafficlight_types[a0]
        self.trafficlight[[1,3,5,7]] = trafficlight_types[a1]

    def create_car(self):
        '''
        if self.spawn_car != 2:
            self.spawn_car = self.spawn_car + 1
            return
        self.random_table_pos = (self.random_table_pos + 1) % 1000
        if self.random_table[self.random_table_pos]:
            self.spawn_car = 0
'''
        self.cars_buf[self.carbuf_pointer, :-1] = self.route[np.random.randint(0,30)]
        self.cars_buf[self.carbuf_pointer, -1] = (0,-1)
        line_0_0, line_0_1 = self.cars_buf[self.carbuf_pointer, 0]
        head, tail = self.line_ends[line_0_0, line_0_1]
        if self.lines_buf[line_0_0, line_0_1, tail, 2] >= self.LEN_ROAD - 5:
            if self.lines_buf[line_0_0,line_0_1,tail,4] == 0:
                pass
                #print('line_113_self.done=True')
                self.done = True
                # print(line_0_0, line_0_1)
                # for angg in self.lines_buf[line_0_0,line_0_1,:,2]:
                #     print(angg, end = ' ' )
                # print('\n')
#                return
#            self.create_car()
            return
        linebuf_pointer = self.linebuf_pointer[line_0_0, line_0_1]
        if tail != -1:
           # print('age')
            self.lines_buf[line_0_0, line_0_1, tail, 1] = linebuf_pointer

        assert self.lines_buf[line_0_0,line_0_1,linebuf_pointer,5] == 0
        self.lines_buf[line_0_0, line_0_1, linebuf_pointer] = (tail if head != -1 else -1, -1, self.LEN_ROAD, self.carbuf_pointer, 2, 1 if line_0_1 == 1 else -1, 0)

        if head == -1:
            self.line_ends[line_0_0, line_0_1, 0] = linebuf_pointer
        self.line_ends[line_0_0, line_0_1, 1] = linebuf_pointer
        hhh, ttt = self.line_ends[line_0_0, line_0_1]; assert ~((hhh == -1)^(ttt == -1))
        self.linebuf_pointer[line_0_0, line_0_1] = (linebuf_pointer + 1) % self.len_lines_buf
        self.carbuf_pointer = (self.carbuf_pointer + 1) % self.len_cars_buf

    def move_car(self):
        for i in range(8):
            for j in range(4):
                aa,bb = self.line_ends[i,j]
                if (aa != -1 and self.lines_buf[i,j,aa,5]==0 )or( bb != -1  and self.lines_buf[i,j,bb,5]==0):
                    print(i,j,aa,bb,'i,j,aa,bb')
                    print('fucked')
                    assert 0
                    '''
                    a = self.lines_buf[i,j,k,3]
                    s = self.cars_buf[a,-1,0]
                    try:
                        assert self.cars_buf[a,s,0] == i and self.cars_buf[a,s,1] == j
                    except:
                        print(s,a,self.cars_buf[a,s], i,j,'s,id,_,_,i,j')
                        print(self.cars_buf[a])
#                        assert 0
                    '''
######################################################
        for i in range(8):            
            for j in range(4):
                queue_len = 0
                waittime_sum = 0

                head, tail = self.line_ends[i,j]
                try:
                    assert ~((head == -1) ^ (tail == -1))
                except:
                    print(i,j)
                    print(head, tail)
                    assert 0
                if head != -1:
                    mask_is_valid = self.lines_buf[...,5] != 0
                    mask_is_valid = mask_is_valid[i,j]
                    pos = self.lines_buf[...,2]
                    pos = pos[i,j]
                    speed = self.lines_buf[...,4]
                    speed = speed[i,j]
                    pos[mask_is_valid] -= speed[mask_is_valid]
                    
                    mask_is_valid = self.lines_buf[...,5] != 0
                    mask_is_in_queue = (self.lines_buf[..., 2] < self.LEN_ROAD/2) & (self.lines_buf[..., 2] >= 0) & (mask_is_valid)
                    mask_is_in_queue = mask_is_in_queue[i,j]
                    mask_is_valid = mask_is_valid[i,j]   


                    queue_len = mask_is_in_queue.sum()
                    self.lines_buf[i,j,mask_is_in_queue, 6] += 1
                    waittime_sum = np.sum(self.lines_buf[i,j,mask_is_in_queue, 6])
                    self.queue_length_per_line[i,j] = queue_len
                    self.mean_waittime_per_line[i,j] = waittime_sum / (queue_len + 1)


                    s_plus = speed+1
                    s_minus = speed-1
                    speed_inc = 2*(2 < s_plus) + (s_plus)*(2 >= s_plus)
                    speed_dec = (s_minus)*(0 < s_minus)
                    greenlight = self.trafficlight[i,j]
                    #print(pos[head])
                    if greenlight or pos[head] >= 3:
                        speed[head] = speed_inc[head]
                    elif pos[head] > 0:
                        speed[head] = 1
                    else:
                        speed[head] = speed_dec[head]

                    __index_of_front = self.lines_buf[..., 0]
                    __index_of_front = __index_of_front[i,j]
                    front_pos = pos[__index_of_front]
                    distance = pos - front_pos
                    
                    mask_not_head = (self.lines_buf[i,j,head,3] != self.lines_buf[i,j,:,3])
                    mask_0 = (distance == 2)
                    mask_1 = ((~mask_0) & (distance <= 4))
                    mask_2 = ~(mask_0 | mask_1)

                    speed[(mask_0 & mask_not_head & mask_is_valid)] = 0
                    speed[(mask_1 & mask_not_head & mask_is_valid)] = 1
                    speed[(mask_2 & mask_not_head & mask_is_valid)] = speed_inc[(mask_2 & mask_not_head & mask_is_valid)]


##################################################################
        for i in range(8):
            for j in range(4):
                head, tail = self.line_ends[i,j]
                try:
                    assert ~((head == -1) ^ (tail == -1))
                except:
                    print(head, tail)
                    assert 0
                if head != -1 and self.lines_buf[i,j,head,2] < 0:
                    assert self.lines_buf[i,j,head,5] != 0
                    self.lines_buf[i,j,head,5] = 0
                    self.lines_buf[i,j,head,0] = -1
                    rear = self.lines_buf[i,j,head,1]
                    self.lines_buf[i,j,rear,0] = -1

                    #self.lines_buf[i,j,head,1] = -1
                    head_id = self.lines_buf[i,j,head,3]
                    cur_step = self.cars_buf[head_id, -1, 0]
                    zzz = (cur_step != 4)
                    if ~zzz:
                        print(i,j)
                        print(head, tail)
                        print(head_id, self.cars_buf[head_id,:])
                        print(rear, ': rear')
                        rear_id = self.lines_buf[i,j,rear,3]
                        print(rear_id, self.cars_buf[rear_id,:])
                        for foo_ in range(7):
                            for bar_ in range(self.len_lines_buf):
                                print('%3d' % self.lines_buf[i,j,bar_,foo_], end=' ')
                            print('')
                        print('\n')
                        assert 0
                    cur_step = cur_step + 1

                    self.cars_buf[head_id, -1, 0] = cur_step

                    l0, l1 = self.cars_buf[head_id, cur_step]
                    if l1 != -1:
                        if l1 != 0:
                            linebuf_pointer = self.linebuf_pointer[l0,l1]
                            
                            h, t = self.line_ends[l0, l1]
                            if t != -1:
                                self.lines_buf[l0, l1, t, 1] = linebuf_pointer
                            self.lines_buf[l0, l1, linebuf_pointer] = (t if h != -1 else -1, -1, self.LEN_ROAD, head_id, 1, 1 if l1 == 1 else -1, 0)
                            
                            if h == -1:
                                self.line_ends[l0, l1, 0] = linebuf_pointer
                            
                            self.line_ends[l0, l1, 1] = linebuf_pointer
                            self.linebuf_pointer[l0, l1] = (linebuf_pointer + 1) % self.len_lines_buf
                        # else:
                        #     if l0 in (0,4,6,2):
                        #         self.fail_0 = self.fail_0 + 1
                        #     else:
                        #         self.fail_1 = self.fail_1 + 1
                    #print('ccc')
                    try:
                        assert ~((head == -1) ^ (tail == -1))
                    except:
                        print(head, tail)
                        assert 0
                    
                    if head == tail:
                        tail = -1
                        self.line_ends[i,j,1] = tail

                    assert head != self.lines_buf[i,j,head,1]
                    head = self.lines_buf[i,j,head,1]
#                    print(i,j,l0,l1,head,head_id,cur_step,'i,j,l0,l1,next__,__id,cur_step')
                    asdf = ~((head == -1) ^ (tail == -1))
                    if ~asdf:
                        print(i,j)
                        print(head, tail)
                        for foo_ in range(7):
                            for bar_ in range(self.len_lines_buf):
                                print('%3d' % self.lines_buf[i,j,bar_,foo_], end=' ')
                            print('')
                        print('\n')
                        assert 0

                    if head != -1:
                        self.lines_buf[i,j,head,0] = -1
                    self.line_ends[i,j,0] = head
                    
                #print('ddd')

                #print('eee')
                
    def step(self, action : tuple):
        #print('---')

        self.iter += 1

        self.adjust_trafficlight(action)
        self.create_car()
        self.create_car()
        self.move_car()
        self.create_car()
        self.create_car()
        self.move_car()

        rms_queue_len_0 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[0,4,6,2]])))
        rms_queue_len_1 = np.sqrt(np.mean(np.square(self.queue_length_per_line[[1,5,7,3]])))
        rms_waittime_0 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[0,4,6,2]], dtype=np.int32)))
        rms_waittime_1 = np.sqrt(np.mean(np.square(self.mean_waittime_per_line[[1,5,7,3]], dtype=np.int32)))
        aang_0 = rms_queue_len_0 * rms_waittime_0 + self.fail_0 * 50
        aang_1 = rms_queue_len_1 * rms_waittime_1 + self.fail_1 * 50
        #rwd = 5000 - rwd
        rwd_0 = np.array((2 + (self.ang_0 - aang_0),))/4
        rwd_1 = np.array((2 + (self.ang_1 - aang_1),))/4

        self.ang_0 = aang_0
        self.ang_1 = aang_1

        # rwd_0 = -np.sum(self.queue_length_per_line[[0,4,6,2]]) - np.sum(self.mean_waittime_per_line[[0,4,6,2]]) - self.fail_0 * 50
        # rwd_1 = -np.sum(self.queue_length_per_line[[1,5,7,3]]) - np.sum(self.mean_waittime_per_line[[1,5,7,3]]) - self.fail_0 * 50
        
        if self.fail_0 != 0:
            print('0')
        if self.fail_1 != 0:
            print('1')

        
        if self.iter >= 20000:
            self.done = True

        if self.fail_0 or self.fail_1:
            self.done = True
        self.fail_0 = 0
        self.fail_1 = 0

        ret = (((*self.queue_length_per_line[[0,4,6,2]].flatten(), *self.mean_waittime_per_line[[0,4,6,2]].flatten()),(*(self.queue_length_per_line[[1,5,7,3]].flatten()), *self.mean_waittime_per_line[[1,5,7,3]].flatten())), (rwd_0, rwd_1), (self.done, self.done))


        if self.done:
            self.reset()
            self.iter = 0
        return ret

def main():
    env = Environment()

    iter = 0
    for i in range(200000):
        iter = iter + 1
        # action = np.random.randint(0,16)
        # obs, rwd, done = env.step(action)
        action1 = np.random.randint(0,4)
        action2 = np.random.randint(0,4)

        obs, rwd, done = env.step((action1, action2))
        if iter%100 == 0:
            print(obs)
            print(iter)
        if done[0]:
            print(obs)
            print(iter)
            print(rwd)
            print('---')
            ang = 0
            # if iter < 40:
            #     print('ang')
            #     raise Exception
            iter = 0

if __name__ == '__main__':
    main()
    
