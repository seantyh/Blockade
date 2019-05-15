import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SAMPLE = False
TRUNCATED_SAMPLE = True
SOS_token = -1
model_random_state = np.random.RandomState(21532)  #pylint: disable=no-member
torch.manual_seed(21532)

class Encoder(nn.Module):
    def __init__(self, char_vec, output_size=100):
        super(Encoder, self).__init__()
        vocab_size, emb_dim = char_vec.shape
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab_size-1)
        self.embed.weight = nn.Parameter(torch.from_numpy(char_vec.vectors.astype('float32')))
        self.embed.weight.requires_grad = False
        self.gru = nn.GRU(emb_dim, emb_dim, 1, bidirectional=False)
        self.o2p = nn.Linear(emb_dim, output_size*2)

    def sample(self, mu, logvar, device):
        eps = torch.randn(mu.size()).to(device)
        std = torch.exp(logvar/2.0)
        return mu + eps * std

    def forward(self, input, device):
        embedded = self.embed(input)        
        output, hidden = self.gru(embedded, None)
        output = output[-1]
        
        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar, device)
        return mu, logvar, z

class Decoder(nn.Module):
    def __init__(self, z_size, char_vec, n_conditions,
                 condition_size, hidden_size, output_size, word_dropout=1.):
        super(Decoder, self).__init__()
        
        self.word_dropout = word_dropout
        self.UNK_token = char_vec.stoi["<UNK>"]
        input_size = z_size + condition_size
        vocab_size, emb_dim = char_vec.shape
        hidden_size = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab_size-1)        
        self.embed.weight = nn.Parameter(torch.from_numpy(char_vec.vectors.astype('float32')))
        self.embed.weight.requires_grad = False
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, 1)
        self.i2h = nn.Linear(input_size, hidden_size)
        if n_conditions > 0 and condition_size > 0 and n_conditions != condition_size:
            self.c2h = nn.Linear(n_conditions, condition_size)
        
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)

    def sample(self, output, temperature, device, 
                max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        if max_sample:
            top_i = output.data.topk(1)[1].item()
        else:
            # sample from the network as a multionmial distribution
            if trunc_sample:
                # sample from top k values only
                k = 10
                new_output = torch.empty_like(output).fill_(float('-inf'))
                top_v, top_i = output.data.topk(k)
                new_output.data.scatter_(1, top_i, top_v)
                output = new_output
            
            output_dist = output.data.view(-1).div(temperature).exp()
            if len(torch.nonzero(output_dist)) > 0:
                top_i = torch.multinomial(output_dist, 1)[0]
            else:
                print("[WARNING] output_dist is all zeros")
            
            input_vec = torch.LongTensor([top_i]).to(device)
            return input_vec, top_i

    def forward(self, z, condition, inputs, temperature, device):
        n_steps = inputs.size(0)
        outputs = torch.zeros(n_steps, 1, self.output_size).to(device)

        input_vec = torch.LongTensor([SOS_token]).to(device)
        if condition is None:
            decode_embed = z
        else:
            if hasattr(self, 'c2h'):
                squashed_condition = self.c2h(condition)
                decode_embed = torch.cat([z, squashed_condition], 1)
            else:
                decode_embed = torch.cat([z, condition], 1)
        
        # original code has an option of multiple layers along first dimension
        hidden = self.i2h(decode_embed).unsqueeze(0).repeat(1, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, decode_embed, input_vec, hidden)
            outputs[i] = output

            use_word_dropout = model_random_state.rand() < self.word_dropout
            if use_word_dropout and i < (n_steps-1):
                unk_input = torch.LongTensor([self.UNK_token]).to(device)
                input_vec = unk_input
                continue
            
            use_teacher_forcing = model_random_state.rand() < temperature
            if use_teacher_forcing:
                input_vec = inputs[i]
            else:
                input_vec, top_i = self.sample(output, temperature, device, max_sample=True)
            
            if input_vec.dim() == 0:
                input_vec = input_vec.unsqueeze(0)
            
        return outputs.squeeze(1)

    def step(self, s, decode_embed, input_vec, hidden):
        inputx = F.relu(self.embed(input_vec))
        inputx = torch.cat((inputx, decode_embed), 1)
        inputx = input_vec.unsqueeze(0)
        output, hidden = self.gru(inputx, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, decode_embed), 1)
        output = self.out(output)

        return output, hidden

class VAE(nn.Module):
    def __init__(self, char_vec, encoder, decoder, n_steps=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
        self.register_buffer('kld_max', torch.tensor(0, dtype=torch.float))
        self.register_buffer('kld_weight', torch.tensor(0, dtype=torch.float))

        if n_steps is not None:
            self.register_buffer('kld_inc', torch.tensor(
                (self.kld_max - self.kld_weight) / (n_steps//2), dtype=torch.float))
        else:
            self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))
        
        
        def forward(self, inputs, targets, condition, device, temperature=1.0):
            mu, logvar, z = self.encoder(inputs, device)
            decoded = self.decoder(z, condition, targets, temperature, device)
            return mu, logvar, z, decoded

