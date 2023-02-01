"""Refs:
        https://github.com/karpathy/nanoGPT/blob/master/train.py
"""
from pathlib import Path


# pre-training
PRETRAIN = True
# model
EMB_DIM = 256
HEADS = 8
NUM_LAYERS = 12
CONTEXT = 512
DROPOUT = 0.0
# logging
LOGDIR = Path('./data/runs')
LOAD = LOGDIR/'ckpt.pt'  # or None
PRINT_INTERVAL = 10
# dataset
DATASET = './data/codeparrot_train.pkl'
CACHE_DIR = './data/cache/codeparrot_train'
EVALSET = './data/codeparrot_test.pkl'
EVAL_CACHE_DIR = './data/cache/codeparrot_test'
# training
BATCH_SIZE = 3
GRAD_ACC_STEPS = 6
MAX_ITERS = 100_000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
GRADIENT_CLIP = None
# adamw optimizer
LR = 6e-4
WEIGHT_DECAY = 1e-2
# learning rate decay settings
DECAY_LR = True
WARMUP_ITERS = 2000
LR_DECAY_ITERS = MAX_ITERS
MIN_LR = LR/10