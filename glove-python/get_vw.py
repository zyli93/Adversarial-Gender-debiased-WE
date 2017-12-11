from __future__ import print_function
import time
from adverserial import Adversarial_FL
import  os

import sys

try:
    import ujson as json
except ImportError:
    import json

import glove_train as GT
from glove import Glove
from glove import Corpus
from operator import itemgetter
import numpy as np

if __name__ == "__main__":
    words = open('word_embdding_data/vectors.words.vocab','r').readlines()
    word_lists = [x.strip() for x in words]
    print(len(word_lists))
    embedding_size = 100
    word2ind = Corpus().load('corpus.model').dictionary
    word_list_ids = [word2ind[w] for w in word_lists]
    glove = Glove.load('glove_originial.model')
    word_embeddings = GT.transform_embedding(word2ind, glove, embedding_size)
    np.savetxt('word_embdding_data/vectors.words', word_embeddings[word_list_ids,:])
    print(len(word_list_ids))
    # with open('vectors.words','w') as f:
    #     for w in word_embeddings:
    #         f.write(w)
    #         f.write('\n')
    
    # sorted_w = sorted(word2ind.items(), key=itemgetter(1))
    # print(len(sorted_w))
    # with open('vectors.words.vocab', 'w') as f:
    #     for k in sorted_w:
    #         f.write(k[0])
    #         f.write('\n')
