import os
import time
from pathlib import Path

import numpy as np
import torch

from rnn import CharRNN
from data import WordSet
from utils import sample

__author__ = "__Girish_Hegde__"


HIDDEN_SIZE = 512
NUM_LAYERS = 3
LOAD = Path('./data/runs/checkpoint_990.pt')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wordset = WordSet('./data/sanskrit_words.txt', DEVICE)
int2char, char2int, embeddings = wordset.int2char, wordset.char2int, wordset.embeddings

net = CharRNN(wordset.vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
load = torch.load(LOAD)
net.load_state_dict(load['state_dict'])

inp = 'ind'
prediction = sample(net, int2char, char2int, embeddings, top_k=1, prime=inp, max_size=12, device=DEVICE)
print(f'input: {inp}, prediction: {prediction}')