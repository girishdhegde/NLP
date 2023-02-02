from pathlib import Path


# model
EMB_DIM = 256
HEADS = 8
NUM_LAYERS = 12
CONTEXT = 512
DROPOUT = 0.1
# logging
LOGDIR = Path('./data/code_finetuning')
LOAD = Path('./data/runs/best.pt')
PRINT_INTERVAL = 10
# dataset
DATASET = './data/cache/codeparrot_train/dataset.pkl'
EVALSET = './data/cache/codeparrot_test/dataset.pkl'
# training
BATCH_SIZE = 3
GRAD_ACC_STEPS = 6
EVAL_INTERVAL = 500
MAX_EPOCHS = 3
# adamw optimizer
LR = 1e-5
WEIGHT_DECAY = 1e-2
# learning rate decay settings
DECAY_LR = False