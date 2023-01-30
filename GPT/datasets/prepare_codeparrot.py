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

np.random.seed(108)

def process():
    que_pretrain, sol_pretrain, que_finetune, sol_finetune = [], [], [], []
    total= 0

    for i, example in tqdm(enumerate(codeparrot), total=len(codeparrot)):
        if (i == 2835) and (split == 'train'): continue
        solutions = example['solutions']
        if not len(solutions): continue
        question = example['question']
        solutions = json.loads(solutions)

        que_enc = tokenizer.encode(question)
        max_len = block_size - len(que_enc)

        que_added = False
        for sol in solutions:
            total += 1
            sol = sol.replace("    ", "\t")
            sol_enc = tokenizer.encode(sol)
            if len(sol_enc) < max_len:
                if not que_added:
                    que_added = True
                    que_finetune.append(question)
                    sol_finetune.append([sol])
                else:
                    sol_finetune[-1].append(sol)
            else:
                sol_pretrain.append(sol)
        que_pretrain.append(question)


    print(f'{total = }, que pretrain = {len(que_pretrain)}, sol pretrain = {len(sol_pretrain)}, finetune = {len(que_finetune)}')
    return que_pretrain, sol_pretrain, que_finetune, sol_finetune


split = 'train'  # or 'test'
outdir = '../data/'
block_size = 512 - 10  # -10 for other task specific and other special tokens if any required
test_percentage = 10

codeparrot = load_dataset("codeparrot/apps", split=split)
print(f'dataset info: {codeparrot}')
tokenizer = tiktoken.get_encoding("gpt2")

que_pretrain, sol_pretrain, que_finetune_train, sol_finetune_train = process()
pretrain = que_pretrain + sol_pretrain

split = 'test'  # or 'test'

codeparrot = load_dataset("codeparrot/apps", split=split)
print(f'dataset info: {codeparrot}')
tokenizer = tiktoken.get_encoding("gpt2")

que_pretrain, sol_pretrain, que_finetune_test, sol_finetune_test = process()
pretrain = pretrain + que_pretrain + sol_pretrain



outdir = Path(outdir)
outdir.mkdir(exist_ok=True, parents=True)
test_mask = np.zeros(len(pretrain), bool)
test_mask[np.random.uniform(0, 1, size=len(pretrain)) < (test_percentage/100)] = True
pretrain = np.array(pretrain, dtype=object)
with (outdir/f'codeparrot_train.pkl').open('wb') as fp:
    pickle.dump(pretrain[np.logical_not(test_mask)], fp)
with (outdir/f'codeparrot_test.pkl').open('wb') as fp:
    pickle.dump(pretrain[test_mask], fp)

que_finetune_train = np.array(que_finetune_train + que_finetune_test, dtype=object)
sol_finetune_train = np.array(sol_finetune_train + sol_finetune_test, dtype=object)
test_mask = np.zeros(len(que_finetune_train), bool)
test_mask[
    np.random.uniform(0, 1, size=len(que_finetune_train)) < (test_percentage/100)
] = True
codeparrot_finetune_train = {
    'questions':que_finetune_train[np.logical_not(test_mask)],
    'solutions':sol_finetune_train[np.logical_not(test_mask)],
}
codeparrot_finetune_test = {
    'questions':que_finetune_train[test_mask],
    'solutions':sol_finetune_train[test_mask],
}

(outdir/'cache'/'codeparrot_finetune_train').mkdir(exist_ok=True, parents=True)
with (outdir/'cache'/'codeparrot_finetune_train'/'dataset.pkl').open('wb') as fp:
    pickle.dump(codeparrot_finetune_train, fp)

(outdir/'cache'/'codeparrot_finetune_test').mkdir(exist_ok=True, parents=True)
with (outdir/'cache'/'codeparrot_finetune_test'/'dataset.pkl').open('wb') as fp:
    pickle.dump(codeparrot_finetune_test, fp)