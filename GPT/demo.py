from pathlib import Path

import torch

from gpt import GPT
from data import BPETokenizer
from utils import sample


__author__ = "__Girish_Hegde__"


CKPT = Path('./data/wikitext_runs/best.pt')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
