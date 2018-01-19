from __future__ import print_function
from adverserial import Adversarial_FL
import sys, os

try:
    import ujson as json
except ImportError:
    import json

import glove_train as GT
from glove import Glove
from glove import Corpus

"""
Adversarial Gender Unbiased Word Embedding

Jieyu Zhao, Yichao Zhou, Zeyu Li
CS269 Course Project

"""

def train_glove(file_name, epoch_num, parallel_num, retrain, dimension, dictionary={}, iter_num=0):
    return GT.training_process(create=file_name, 
                               train=epoch_num, 
                               parallel=parallel_num, 
                               retrain=retrain, 
                               embedding_dimension = dimension, 
                               word_dictionary = dictionary,
                               iteration_num=iter_num)


if __name__ == "__main__":

    if len(sys.argv) != 1 + 1:
        print("Usage"
              "\tpython {} <number of iterations>".format(sys.argv[0]),
              file=sys.stderr)
        sys.exit()

    iteration_num = int(sys.argv[1])
    print("Setting number of iteration as {}".format(iteration_num))

    opt = int(sys.argv[1])
    DATA_FOLDER = '../data/'
    DATA_FILE = DATA_FOLDER + 'text8'
    if not os.path.exists(DATA_FILE):
        print("Please create ../data/ dir and put text8 corpus in that dir",
              file=sys.stderr)

    # Setting parameters
    embedding_size = 100

    # Loading or training the initial model
    print("\n - ITER 0: Training the first glove ....")
    word2ind = Corpus().load('corpus.model').dictionary
    glove = Glove.load('glove0.model')
    word_embeddings = GT.transform_embedding(word2ind, glove, embedding_size)


    # Loading gender words
    print("Loading the gender related words ...")
    gw_file = DATA_FOLDER + 'gender_specific_full.json'
    if not os.path.exists(gw_file):
        print("Please in include {}".format(gw_file),
              file=sys.stderr)
    gw = json.load(open(gw_file))
    gw_index = [word2ind[item] if item in word2ind else -1
                for item in gw]

    for i in range(iteration_num):
        print("ITER "+str(i)+": Training Adversarial FL ...")
        # train the Adversarial_FL
        batch_size = 64
        lr = 1e-3

        adv = Adversarial_FL(word_embeddings=word_embeddings,
                             embedding_size=embedding_size,
                             gender_words_index=gw_index,
                             word2ind=word2ind,
                             cuda=False)
        print("Training Adversarial ...")
        new_embedding, _ = adv.train(lr=lr,
                                     batch_size=batch_size,
                                     # epochs=10, TODO: Decide the # of tranining epoches
                                     epochs=1,
                                     gender_words_index=gw_index)

        i += 1
        print("ITER "+str(i)+": Training glove ....")

        word2ind, glove = GT.training_process(create=DATA_FILE,
                                              train=2,
                                              parallel=4,
                                              retrain=0,
                                              embedding_dimension=embedding_size,
                                              word_dictionary=new_embedding,
                                              iteration_num=i)
        word_embeddings = GT.transform_embedding(word2ind, glove, embedding_size)

        print("Done!")
