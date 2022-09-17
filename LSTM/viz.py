"""
Refs:
    https://youtu.be/PaCmpygFfXo
"""


from pathlib import Path

import torch

from rnn import CharRNN
from utils import get_BiGram, BiGram_viz


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/runs/best.pt')
DEVICE = torch.device('cpu')

ckpt = torch.load(CKPT, map_location=DEVICE)
net = CharRNN(ckpt['tokens'], ckpt['HIDDEN_SIZE'], ckpt['NUM_LAYERS'])
net.load_state_dict(ckpt['state_dict'])
net = net.to(DEVICE)
print('Checkpoint loaded successfully ...')

bigram = get_BiGram(net, ckpt['int2char'])
BiGram_viz(bigram, ckpt['int2char'], filename='./data/viz/bigram.png')

