import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os, sys, time
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

try:
    import ujson as json
except ImportError:
    import json

"""
Class of Discriminator
"""
class Discriminator(nn.Module):
    def __init__(self,  size):
        super(Discriminator, self).__init__()
        self.size = size

        self.linear1 = nn.Linear(self.size, self.size, bias=True)
        self.linear2 = nn.Linear(self.size, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.lrelu(self.linear1(x))
        score = self.sigmoid(self.linear2(x))
        return score


"""
Class of Encoder,
    Fully connected layer: emb_size + 1 to emb_size
"""
class Encoder(nn.Module):
    def __init__(self, emb_size):
        super(Encoder, self).__init__()
        # Here the embedding is the s augmented Variable
        self.input_size = emb_size
        self.output_size = self.input_size - 1
        self.linear = nn.Linear(in_features=self.input_size,
                                out_features=self.output_size,
                                bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, s_emb):
        return self.lrelu(self.linear(s_emb))


"""
Class of Adversarial Learning
"""
class Adversarial_FL(object):
    def __init__(self, word_embeddings, embedding_size, gender_words_index, word2ind, cuda=True):
        self.word_embeddings = word_embeddings
        self.word2ind = word2ind
        self.cuda = cuda
        self.embedding_size = embedding_size
        self.gender_index = [x for x in gender_words_index if x >= 0]

        if self.cuda:
            self.D.cuda()

    def train(self, lr=1e-3, batch_size=50, epochs=1000, gender_words_index=None):
        # Do all the settings
        E = Encoder( emb_size=self.embedding_size + 1).cuda()
        D = Discriminator( size=self.embedding_size).cuda()
        criterion = nn.BCELoss()
        d_optimizer = optim.Adam(D.parameters()) # TODO: set lr, betas
        e_optimizer = optim.Adam(E.parameters()) # TODO: set lr, betas

        # TODO: Add following stmt when using cuda
        # if self.cuda:
        #     s = s.cuda()
        #     criterion = criterion.cuda()

        dtype = t.cuda.FloatTensor
        # old_avg_loss = 0.
        for epoch in range(epochs):
            # avg_loss = 0.
            # Load data (word embedding)
            batch_ids = DataLoader(range(len(self.word_embeddings)), batch_size, shuffle=True, num_workers=1)
            for i, ids in enumerate(batch_ids):
                E.zero_grad()
                D.zero_grad()
                true_s = Variable(t.Tensor([1 if i in gender_words_index else 0 for i in ids]).view(len(ids), -1))
                e_input = Variable(
                    t.from_numpy(np.append(self.word_embeddings[ids], true_s.data, axis = 1)).type(dtype),
                    requires_grad=True)
                true_s = true_s.cuda()
                e_input = e_input.cuda()
                encode_emb = E(e_input) # Encode the embedding with gender signal `s`
                d_decision = D(encode_emb) # Use discriminator to discriminate the value
                loss = criterion(d_decision, true_s) / batch_size
                loss.backward()
                d_optimizer.step()
                e_optimizer.step()

        # Re-construct embeddings
        res =  {}
        for i, emb in enumerate(self.word_embeddings):
            true_s = 1 if i in gender_words_index else 0
            e_input = Variable(t.from_numpy(np.append(emb, [true_s])).type(dtype))
            e_input = e_input.cuda()
            encode_emb = E(e_input)
            res[i] = encode_emb.data

        return res, res[0].shape[0]

    def get_s(self):
        """
        Get $s$ from 14 definitional pairs of words.

        Take all differences between the embeddings of the word pairs.
        Use PCA to extract the 1st (i.e. largest) components as $s$.
        """
        def_pairs = json.load(open('../data/definitional_pairs.json'))
        def_pairs_ind = [[self.word2ind[pair[0]], self.word2ind[pair[1]]]
                         for pair in def_pairs if pair[0] in self.word2ind and pair[1] in self.word2ind]
        diff = [self.word_embeddings[ind_pair[0]] - self.word_embeddings[ind_pair[1]]
                for ind_pair in def_pairs_ind]
        pca = PCA(n_components=1)
        pca.fit(np.array(diff))
        return pca.components_.flatten() # This returns the 1st principle component as $s$.
