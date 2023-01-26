from pathlib import Path

import torch

from gpt import GPT
from data import BPETokenizer
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/wikitext_runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location='cpu')
net = GPT(**ckpt['net']['config'])
net.load_state_dict(ckpt['net']['state_dict'])
n_tasks = ckpt['kwargs']['n_tasks'] if 'n_tasks' in ckpt['kwargs'] else 0
tokenizer = BPETokenizer(n_tasks)


prompt = "The"
prediction = sample(
    prompt, net, tokenizer,
    max_new_tokens=512, temperature=1.0, top_k=None,
    device=DEVICE
)
print(prediction)