import sys
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:]

arr = []
with open('logs/'+args[0], 'r') as log:
    titles = list(map(str.strip, log.readline().split(":")))
    while True:
        line = log.readline()
        if not line: break
        arr.append(list(map(str.strip, line.split(":"))))
arr = np.array(arr)

print(args[1:])
args_len = len(args[1:])
for iter, arg in zip(range(args_len), args[1:]):
    sel = int(arg)
        
    arr_ = np.array(list(map(float, arr[:,sel])))
    ten_lst = arr_[:10]
    _len = len(arr_)
    pointer = _len - 1
    result = []
    for i in range(10,_len):
        result.append(ten_lst.mean())
        pointer = (pointer + 1) % 10
        ten_lst[pointer] = arr_[i]
    asdf = args_len * 100 + 10 + (iter + 1)
    plt.subplot(asdf)
    plt.plot(result[10:])
    plt.title(titles[sel])

plt.show()