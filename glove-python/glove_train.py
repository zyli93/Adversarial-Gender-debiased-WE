"""
Joey
12/07/2017
This .py starts Glove with 2 mode: train & retrain

"""


from __future__ import print_function
import argparse
import pprint
import gensim

from glove import Glove
from glove import Corpus
import numpy as np


def check_random_state(seed):
    """ Turn seed into a np.random.RandomState instance.

        This is a copy of the check_random_state function in sklearn
        in order to avoid outside dependencies.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def read_corpus(filename):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    with open(filename, 'r') as datafile:
        for line in datafile:
            yield line.lower().translate(None, delchars).split(' ')


def transform_embedding(dictionary, model, embedding_dimension):

    random_state = check_random_state(None)
    embdding_vector = ((random_state.rand(len(dictionary),embedding_dimension) - 0.5)/ embedding_dimension)
    count = 0

    for key,value in dictionary.iteritems():
        try:
            assert len(model.word_vectors[int(value)]) == embedding_dimension
            embdding_vector[int(value)] = model.word_vectors[int(value)]
        except:
            count += 1
            pass
    print (count)
    return embdding_vector


def training_process(create=None, train=0, parallel=1, retrain=False, embedding_dimension = 100, word_dictionary = {}, iteration_num=0):
    

    if create:
        # Build the corpus dictionary and the cooccurrence matrix.
        print('Pre-processing corpus')

        get_data = read_corpus

        if not retrain:
            corpus_model = Corpus()
            corpus_model.fit(get_data(create), window=10)
            corpus_model.save('corpus.model')
        else:
            corpus_model = Corpus().load('corpus.model')

        print('--Dict size: %s' % len(corpus_model.dictionary))
        print('--Collocations: %s' % corpus_model.matrix.nnz)

    if train:
        # Train the GloVe model and save it to disk.

        if not create:
            # Try to load a corpus from disk.
            print('--Reading corpus statistics')
            corpus_model = Corpus.load('corpus.model')

            print('--Dict size: %s' % len(corpus_model.dictionary))
            print('--Collocations: %s' % corpus_model.matrix.nnz)

        glove = Glove(no_components=embedding_dimension, learning_rate=0.05)

        if retrain:
            loaded_model = word_dictionary
            embedding_vector = transform_embedding(corpus_model.dictionary, loaded_model, embedding_dimension)
            print (embedding_vector.shape) #(253855, 100)

            glove.fit(corpus_model.matrix, 
                        init_embedding = embedding_vector, 
                        epochs=int(train),
                        no_threads=parallel, 
                        verbose=True)
        else:
            glove.fit(corpus_model.matrix,  
                        epochs=int(train),
                        no_threads=parallel, 
                        verbose=True)

        glove.add_dictionary(corpus_model.dictionary)
        glove.save('glove'+str(iteration_num)+'.model')

        return corpus_model.dictionary, glove