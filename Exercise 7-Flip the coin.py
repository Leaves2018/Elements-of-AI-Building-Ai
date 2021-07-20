import numpy as np

def generate(p1):
    # change this so that it generates 10000 random zeros and ones
    # where the probability of one is p1
    seq = np.random.choice([0, 1], p=[1-p1, p1], size=10000)
    return seq

def count(seq):
    # insert code to return the number of occurrences of 11111 in the sequence
    ans = 0
    i = 0
    while i < len(seq):
        cnt = 0
        while i < len(seq) and seq[i] == 1:
            cnt += 1
            if cnt >= 5:
                ans += 1
            i += 1
        i += 1
    return ans

def main(p1):
    seq = generate(p1)
    return count(seq)

print(main(2/3))