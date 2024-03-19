import torch
import torch.nn as nn

class EmbNet(nn.Module):
    #static default variables
    WORD_DIM = 50
    NUM_CLASSES = 5

    def __init__(self, word_vectors, in_dim=5*WORD_DIM, hidden_dim=WORD_DIM, out_dim=NUM_CLASSES):
        super(EmbNet, self).__init__()

        self._embed = nn.Embedding.from_pretrained(word_vectors, freeze=False)
        self._lin = nn.Linear(in_dim, hidden_dim)
        self._act = nn.Sigmoid()
        self._pred = nn.Linear(hidden_dim, out_dim, bias=False)

    #x is batchSize x windowLength
    def forward(self, x):
        #maps batchSize x windowLength -> batchSize x windowLength x wordDim
        emb = self._embed(x)

        #flattens dimensionality starting at 1st index: (batchSize x windowLength*wordDim)
        win_emb = torch.flatten(emb, start_dim=1)

        z = self._lin(win_emb)
        h = self._act(z)
        y_hat = self._pred(h)

        return y_hat