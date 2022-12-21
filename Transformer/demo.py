from pathlib import Path

import torch
from spacy.lang.en import English

from transformer import Transformer
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/eng_hindi/runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location=DEVICE)
kwargs, state_dict = ckpt['net'].values()
epoch, loss, best = ckpt['training'].values()
in_int2tk, out_int2tk, start_token, end_token, pad_token, ukn_token = ckpt['dataset'].values()

net = Transformer(**kwargs)
net.load_state_dict(state_dict)
net = net.to(DEVICE)
tokenizer = English()

inp = 'I am going home'
pred = sample(
    inp, net, tokenizer,
    in_int2tk, out_int2tk,
    start_token, end_token,
    pad_token, ukn_token,
    top_k=1, max_size=100,
    device=DEVICE,
)
print(pred)