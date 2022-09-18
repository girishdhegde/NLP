from pathlib import Path

import torch

from lstm import WordLSTM
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location=DEVICE)
net = WordLSTM(ckpt['VOCAB_SIZE'], ckpt['EMBEDDING_DIM'], ckpt['HIDDEN_SIZE'], ckpt['NUM_LAYERS'])
net.load_state_dict(ckpt['state_dict'])
net = net.to(DEVICE)

prime = 'When I used to read fairy-tales, I fancied that kind of thing never happened'
pred = sample(net, ckpt['int2token'], top_k=3, prime=prime, max_size=1000, device='cpu')
print(pred)