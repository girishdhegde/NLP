from pathlib import Path

import torch

from rnn import CharRNN
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location=DEVICE)
net = CharRNN(ckpt['tokens'], ckpt['HIDDEN_SIZE'], ckpt['NUM_LAYERs'])
net.load_state_dict(ckpt['state_dict'])
net = net.to(DEVICE)
print('Checkpoint loaded successfully ...')

pred = sample(net, ckpt['int2char'], top_k=3, prime=None, max_size=20, device='cpu', eow='<E>')
print(f'prediction: {pred}')


# TODO: BiGram viz, word2class validation and collate func, code standardization, citation
