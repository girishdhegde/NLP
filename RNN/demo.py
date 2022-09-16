"""
Refs:
    https://www.kaggle.com/code/ashukr/char-rnn/notebook
"""


from pathlib import Path

import torch

from rnn import CharRNN
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location=DEVICE)
net = CharRNN(ckpt['tokens'], ckpt['HIDDEN_SIZE'], ckpt['NUM_LAYERS'])
net.load_state_dict(ckpt['state_dict'])
net = net.to(DEVICE)
print('Checkpoint loaded successfully ...')

pred = sample(net, ckpt['int2char'], top_k=3, prime=None, max_size=20, device=DEVICE, eow='<E>')
print(f'prediction: {pred}')


# TODO: BiGram viz