"""
Refs:
    https://huggingface.co/datasets/codeparrot/apps
"""

import json
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
from datasets import load_dataset
import tiktoken  # openai gpt2 bpe tokenizer


__author__ = "__Girish_Hegde__"


split = 'train'  # or 'test'
outdir = '../../data/'
block_size = 512 - 10  # -10 for other task specific and other special tokens if any required.

codeparrot = load_dataset("codeparrot/apps", split=split)
print(f'dataset info: {codeparrot}')
tokenizer = tiktoken.get_encoding("gpt2")

sol_pretrain, que_finetune, sol_finetune, finetune_id = [], [], [], []
total, fid = 0, 0

for i, example in tqdm(enumerate(codeparrot), total=len(codeparrot)):
    if i == 2835: continue
    solutions = example['solutions']
    if not len(solutions): continue
    question = example['question']
    solutions = json.loads(solutions)

    que_enc = tokenizer.encode(question)
    max_len = block_size - len(que_enc)

    finetune = 0
    for sol in solutions:
        total += 1
        sol = sol.replace("    ", "\t")
        sol_enc = tokenizer.encode(sol)
        if len(sol_enc) < max_len:
            finetune = 1
            que_finetune.append(question)
            sol_finetune.append(sol)
            finetune_id.append(fid)
        else:
            sol_pretrain.append(sol)
    fid += finetune

print(f'{total = }, total pretrain = {len(sol_pretrain)}, total finetune = {len(sol_finetune)}, unique finetune = {fid}')

outdir = Path(outdir)
outdir.mkdir(exist_ok=True, parents=True)
if split == 'train':
    with (outdir/f'codeparrot_train.pkl').open('wb') as fp:
        pickle.dump(sol_pretrain, fp)

codeparrot_finetune = {
    'split':split,
    'unique':fid,
    'ids':finetune_id,
    'questions':que_finetune,
    'solutions':sol_finetune,
}
with (outdir/f'codeparrot_{split}_finetune.pkl').open('wb') as fp:
    pickle.dump(codeparrot_finetune, fp)

