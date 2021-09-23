import sys
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:]
sel = int(args[1])
arr = []
with open('logs/'+args[0], 'r') as log:
    title = log.readline().split(":")[sel].strip()
    while True:
        line = log.readline()
        if not line: break
        arr.append(list(map(str.strip, line.split(":"))))

arr = np.array(arr)
arr = np.array(list(map(float, arr[:,sel])))
ten_lst = arr[:10]
_len = len(arr)
pointer = _len - 1
result = []
for i in range(10,_len):
    result.append(ten_lst.mean())
    pointer = (pointer + 1) % 10
    ten_lst[pointer] = arr[i]
plt.plot(result[10:])
plt.title(title)
plt.show()