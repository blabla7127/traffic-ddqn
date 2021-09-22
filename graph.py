import sys
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:]
arr = []
with open('logs/'+args[0], 'r') as log:
    log.readline()
    while True:
        line = log.readline()
        if not line: break
        arr.append(list(map(str.strip, line.split(":"))))

arr = np.array(arr)
arr1 = list(map(float, arr[10:,2]))
arr2 = np.array(list(map(float, arr[:,-3])))
ten_lst = arr2[:10]
_len = len(arr2)
pointer = _len - 1
result = []
for i in range(10,_len):
    result.append(ten_lst.mean())
    pointer = (pointer + 1) % 10
    ten_lst[pointer] = arr2[i]
plt.plot(result[10:])
plt.show()
#plt.plot(arr1)
#plt.show()