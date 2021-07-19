# These codes are partly based on the following link:
# https://github.com/gojo5t5/elements-of-ai-building-ai/blob/main/BuildingAI/18_tf-idf.py

import math
import numpy as np

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

def main(text):
    # tasks your code should perform:

    # 1. split the text into words, and get a list of unique words that appear in it
    # a short one-liner to separate the text into sentences (with words lower-cased to make words equal 
    # despite casing) can be done with 
    docs = [line.lower().split() for line in text.split('\n')]
    word_list = list(set([word for doc in docs for word in doc]))
    N = len(docs)
    M = len(word_list)

    # 2. go over each unique word and calculate its term frequency, and its document frequency
    tf = dict()
    idf = dict()
    for word in word_list:
        tf[word] = [doc.count(word)/len(doc) for doc in docs]
        idf[word] = sum([word in doc for doc in docs])/N    

    # 3. after you have your term frequencies and document frequencies, go over each line in the text and 
    # calculate its TF-IDF representation, which will be a vector
    tf_idf_vector = np.empty((N, M), dtype=float)
    for i, doc in enumerate(docs):
        for j, word in enumerate(word_list):
            tf_idf_vector[i][j] = tf[word][i] * (0 if idf[word] == 0 else math.log(1/idf[word], 10))

    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.
    dist = np.empty((N, N), dtype=np.float)
    for i in range(N):
        for j in range(N):
            dist[i][j] = np.inf if i == j else sum(abs(tf_idf_vector[j] - tf_idf_vector[i]))

    # print(dist)
    
    print(np.unravel_index(np.argmin(dist), dist.shape))


main(text)