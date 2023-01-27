"""
Refs:
    https://huggingface.co/datasets/wikitext
"""


import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset


__author__ = "__Girish_Hegde__"


split = 'train'  # or 'test'
outdir = '../data/'

wikitext = load_dataset("wikitext", 'wikitext-2-v1')
dataset = wikitext[split]
print(f'dataset info: {dataset}')
dataset = dataset['text']

outdir = Path(outdir)
outdir.mkdir(exist_ok=True, parents=True)
with (outdir/f'wikitext_{split}.pkl').open('wb') as fp:
    pickle.dump(dataset, fp)
