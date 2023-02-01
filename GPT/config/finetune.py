from pathlib import Path


# finetuning
PRETRAIN = False
# model
EMB_DIM = 256
HEADS = 8
NUM_LAYERS = 12
CONTEXT = 512
DROPOUT = 0.0
# logging
LOGDIR = Path('./data/code_finetuning')
LOAD = Path('./data/runs/best.pt')
PRINT_INTERVAL = 10
# dataset
CACHE_DIR = './data/cache/codeparrot_finetune_train'
EVAL_CACHE_DIR = './data/cache/codeparrot_finetune_test'
# training
BATCH_SIZE = 3
GRAD_ACC_STEPS = 6
MAX_ITERS = 10_000
EVAL_INTERVAL = 500
# adamw optimizer
LR = 1e-5
WEIGHT_DECAY = 1e-2
# learning rate decay settings
DECAY_LR = False


