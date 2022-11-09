import numpy as np
import torch

w = torch.Tensor(np.load('W_embedding.npy'))
nlp = torch.Tensor(np.load('test_nlp_trans.npy'))

new_nlp = torch.zeros(nlp.shape[0], nlp.shape[1], 300)

for i in range(len(nlp)):
    ind = nlp[i]
    new_nlp[i] = torch.index_select(w, 0, ind.int())

np.save('test_text.npy', new_nlp.numpy())
