import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from multiprocessing import Pool, current_process


class Tools:
    def __init__(self):
        pass

    def get_co_oc_matrix(self, vocabulary_size, word2ind, corpus_list, context_size = 3):
        corpus_size = len(corpus_list)
        comat = np.zeros((vocabulary_size, vocabulary_size))
        for i in range(corpus_size): #main word
            left_context_ids = [word2ind[ind] for ind in corpus_list[max(0, i - context_size): i]]  #left context
            right_context_ids = [word2ind[ind] for ind in corpus_list[i + 1: min(i + context_size + 1, corpus_size)]] #right context
            ind = word2ind[corpus_list[i]]
            for left_ind, lind in enumerate(left_context_ids):
                comat[ind, lind] += 1./(len(left_context_ids) - left_ind) #symmetrically
            for right_ind, rind in enumerate(right_context_ids):
                comat[ind, rind] += 1./(right_ind + 1)
        return comat

    def get_co_oc_matrix_sparse(self, vocabulary_size, word2ind, corpus_list, context_size = 3, n_proc = 1):
        corpus_len = len(corpus_list)
        chunk_len = int(corpus_len / n_proc + 1)
        chunk_idx = [x for x in range(0, corpus_len, chunk_len)]
        chunk_idx.append(corpus_len)

        pool = Pool(processes=n_proc)
        args = [[corpus_list[chunk_idx[i]:chunk_idx[i+1]], word2ind,
                 vocabulary_size, context_size]
                for i in range(0, len(chunk_idx)-1)]
        results = pool.starmap(func=self.get_co_oc_matrix_workers, iterable=args)
        # print(type(results[0]))
        sparse_matrix = csr_matrix((vocabulary_size, vocabulary_size))
        i = 0
        for res in results:
            print("i = {}".format(i))
            i += 1
            sparse_matrix += res
        return sparse_matrix

    # Get a random batch
    def get_batch(self, vocab_size, batch_size):
        np.random.seed(1)
        in_index = np.random.choice(np.arange(vocab_size), size=batch_size, replace=False)
        out_index = np.random.choice(np.arange(vocab_size), size=batch_size, replace=False)
        return in_index, out_index

    def get_co_oc_matrix_workers(self, corpus_list, word2ind, vocabulary_size, context_size = 3 ):
        corpus_size = len(corpus_list)
        comat = dok_matrix((vocabulary_size, vocabulary_size)) # dok_matrix is the fastest in csc_matrix, csr_matrix, and dok_matrix
        for i in range(corpus_size):
            left_context_ids = [word2ind[ind] for ind in corpus_list[max(0, i - context_size): i]]  #left context
            right_context_ids = [word2ind[ind] for ind in corpus_list[i + 1: min(i + context_size + 1, corpus_size)]] #right context
            ind = word2ind[corpus_list[i]]
            for left_ind, lind in enumerate(left_context_ids):
                comat[ind, lind] += 1./(len(left_context_ids) - left_ind) #symmetrically
            for right_ind, rind in enumerate(right_context_ids):
                comat[ind, rind] += 1./(right_ind + 1)
            if i % 100 == 0:
                print("{} Done {:f}%".format(current_process(), float(i/corpus_size)))
        return csr_matrix(comat)
        # return comat
