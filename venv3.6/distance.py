# import argparse
import numpy as np


# N = 10;          # number of closest words that will be shown

def generate():

    with open('vocab.txt', 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open('vectors.txt', 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term, N):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
#             print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
#             print('Word: %s  Out of dictionary!\n' % term)
            return
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:N]
    b = []
    # print(len(a))

#     print("\n                               Word       Cosine distance\n")
#     print("---------------------------------------------------------\n")
    for x in a:
        t = [ivocab[x], dist[x]]
        b.append(t)
#         print("%35s\t\t%f\n" % (ivocab[x], dist[x]))
    return b


# if __name__ == "__main__":
    
#     W, vocab, ivocab = generate()
#     while True:
#         input_term = raw_input("\nEnter word or sentence (EXIT to break): ")
#         if input_term == 'EXIT':
#             break
#         else:
#             distance(W, vocab, ivocab, input_term)

