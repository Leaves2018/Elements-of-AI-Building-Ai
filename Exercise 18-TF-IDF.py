import numpy as np
import math

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
    word_list = list(set([word for row in docs for word in row]))

    # 2. go over each unique word and calculate its term frequency, and its document frequency
    from collections import defaultdict
    idc_dict = defaultdict(int)
    m = len(docs)
    n = len(word_list)
    tf_idf_vector = np.empty((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            cnt = docs[i].count(word_list[j])
            if cnt > 0:
                idc_dict[word_list[j]] += 1
            tf_idf_vector[i][j] = cnt / len(docs[i])

    # 3. after you have your term frequencies and document frequencies, go over each line in the text and 
    # calculate its TF-IDF representation, which will be a vector
    for i in range(m):
        for j in range(n):
            tf_idf_vector[i][j] *= math.log(m / idc_dict[word_list[i]], 10)

    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.
    dist = np.empty((m, m), dtype=np.float)
    for i in range(m):
        for j in range(m):
            dist[i][j] = np.inf if i == j else sum(abs(tf_idf_vector[j] - tf_idf_vector[i]))

    # print(dist)
    
    print(np.unravel_index(np.argmin(dist), dist.shape))


main(text)