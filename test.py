from typing import final
import numpy as np
import matplotlib.pyplot as plt
initial_epsilon = 1
middle_epsilon = 0.25
final_epsilon = 0.07
epsilon = initial_epsilon
epsilons = []

decay = np.power(middle_epsilon/initial_epsilon, 1/1e6)
decay2 = np.power(final_epsilon/middle_epsilon, 1/3e6)
print(decay)
for i in range(int(7e6)):
    if epsilon > middle_epsilon:
        epsilon *= decay
    elif epsilon > final_epsilon:
        epsilon *= decay2
    epsilons.append(epsilon)
plt.plot(epsilons)
plt.show()