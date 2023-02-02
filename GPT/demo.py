from pathlib import Path

import torch

from gpt import GPT
from data import BPETokenizer
from utils import sample


__author__ = "__Girish_Hegde__"


# =============================================================
# Pre-Trained Model Sampling
# =============================================================
CKPT = Path('./data/runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location='cpu')
net = GPT(**ckpt['net']['config'])
net.load_state_dict(ckpt['net']['state_dict'])
n_tasks = ckpt['kwargs']['n_tasks'] if 'n_tasks' in ckpt['kwargs'] else 0
tokenizer = BPETokenizer(n_tasks)


prompt = "The"
output, prediction = sample(
    prompt, net, tokenizer,
    max_new_tokens=512, temperature=1.0, top_k=None, end_token=None,
    device=DEVICE,
)
print(prediction)



# =============================================================
# Fine-Tuned Model on Code Sampling
# =============================================================
CKPT = Path('./data/code_finetuning/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ckpt = torch.load(CKPT, map_location='cpu')
net = GPT(**ckpt['net']['config'])
net.load_state_dict(ckpt['net']['state_dict'])
n_tasks = ckpt['kwargs']['n_tasks'] if 'n_tasks' in ckpt['kwargs'] else 0
tokenizer = BPETokenizer(n_tasks)
net.eval()
code_token, end_token = ckpt['kwargs']['code_token'], ckpt['kwargs']['end_token']
int2tk = dict(zip(tokenizer._special_tokens.values(), tokenizer._special_tokens.keys()))

prompt = """You are given with 3 numbers x, y, z. Find the largest of 3 numbers given.

-----Input-----
A single line containing space seperated number inputs.

-----Output-----
In a single line containing largest/maximum of x, y, z.

-----Examples-----
Input
1 4 4

Output
4

Input
3 6 3

Output
6
"""

out, prediction = sample(
    prompt + int2tk[code_token], net, tokenizer,
    max_new_tokens=256, temperature=1.0, top_k=3, end_token=end_token,
    device=DEVICE,
)
print(prediction)

# =============================================================
# END
# =============================================================