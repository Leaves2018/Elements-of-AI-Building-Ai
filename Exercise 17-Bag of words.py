import numpy as np

data = [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]]

def find_nearest_pair(data):
    N = len(data)
    data = np.array(data)
    dist = np.empty((N, N), dtype=np.float)
    for i in range(N):
        for j in range(N):
            dist[i][j] = np.inf if i == j else sum(abs(data[j] - data[i]))
    print(np.unravel_index(np.argmin(dist), dist.shape))

find_nearest_pair(data)