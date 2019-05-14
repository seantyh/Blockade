#pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, char_vec):
        super(VAE, self).__init__()
        vocab_size, emb_dim = char_vec.shape
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab_size-1)
        self.embed.weight = nn.Parameter(torch.from_numpy(char_vec.vectors.astype('float32')))
        self.embed.weight.requires_grad = False
        self.gru = nn.GRU(emb_dim, emb_dim, 1, bidirectional=False)
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, input, device):
        embedded = self.embed(input)        
        output, hidden = self.gru(embedded, None)
        return output, hidden