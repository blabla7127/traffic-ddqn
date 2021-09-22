# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import ArtistAnimation

# nx = 162
# ny = 106
# nframe = 64

# fig, ax = plt.subplots(1,1)
# artists = []
# for iframe in range(nframe):
#     nums = np.zeros((ny,nx))
#     nums[0:50,0:50] = 1
#     nums[0:50,56:106] = 1
#     nums[0:50,112:162] = 1
#     nums[56:106,0:50] = 1
#     nums[56:106,56:106] = 1
#     nums[56:106,112:162] = 1
#     nums[50:56,50:56] = 2
#     nums[50:56,106:112] = 2

#     ms = ax.matshow(nums)
#     artists.append([ms])
# ani = ArtistAnimation(fig, artists)
# ani.save('lat.mp4')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig, ax = plt.subplots()
ims=[]

for iternum in range(4):
    title = plt.text(0.5,1.01,iternum, ha="center",va="bottom",color=np.random.rand(3),
                     transform=ax.transAxes, fontsize="large")
    text = ax.text(iternum,iternum,iternum)
    scatter = ax.scatter(np.random.randint(0,10,5), np.random.randint(0,20,5),marker='+')
    ims.append([text,scatter,title,])


ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False,
                              repeat_delay=2000)
plt.show()
