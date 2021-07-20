import math
import numpy as np

x = np.array([4, 3, 0])
c1 = np.array([-.5, .1, .08])
c2 = np.array([-.2, .2, .31])
c3 = np.array([.5, -.1, 2.53])

def sigmoid(z):
    # add your implementation of the sigmoid function here
    return 1 / (1 + np.exp(-z))

# calculate the output of the sigmoid for x with all three coefficients
print(sigmoid(sum(x * c1))) # 0.15446526508353473
print(sigmoid(sum(x * c2))) # 0.45016600268752216
print(sigmoid(sum(x * c3))) # 0.8455347349164652