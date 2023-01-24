import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gpt import GPT
from data import BPETokenizer, PretrainSet
from utils import set_seed, save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"


# =============================================================
# Parameters  (GPT2 small version like params from https://github.com/karpathy/nanoGPT/blob/master/train.py)
# =============================================================
# model
EMB_DIM = 256
HEADS = 8
NUM_LAYERS = 12
VOCAB_SIZE = 50_256
CONTEXT = 512
DROPOUT = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
# logging
LOGDIR = Path('./data/runs')
LOAD = LOGDIR/'best.pt'  # or None
PRINT_INTERVAL = 100
# dataset
DATASET = './GPT/data/wikitext_train.pkl'
CACHE_DIR = './GPT/data/cache/wikitext_train'
EVALSET = './GPT/data/wikitext_train.pkl'
EVAL_CACHE_DIR = './GPT/data/cache/wikitext_train'
# training
BATCH_SIZE = 16
GRAD_ACC_STEPS = 1  # used to simulate larger batch sizes
MAX_ITERS = 600000  # total number of training iterations
EVAL_INTERVAL = 2000
EVAL_ITERS = 200
EVAL_ONLY = False  # if True, script exits right after the first eval
GRADIENT_CLIP = None  # 5
# adamw optimizer
LR = 6e-4  # max learning rate
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95
# learning rate decay settings
DECAY_LR = True  # whether to decay the learning rate
WARMUP_ITERS = 2000  # how many steps to warm up for
LR_DECAY_ITERS = 600000  # should be ~= max_iters per Chinchilla
MIN_LR = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# init
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)
set_seed(108)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True  # optimize backend algorithms

# =============================================================
# Tokenizer, Dataset, Dataloader init
# =============================================================
tokenizer = BPETokenizer(n_tasks=10)
if (Path(CACHE_DIR)/'dataset.pkl').is_file():
    trainset = tokenizer.read_dataset((Path(CACHE_DIR)/'dataset.pkl'))
else:
    trainset = tokenizer.tokenize_dataset(DATASET, CACHE_DIR, True, True)
if (Path(EVAL_CACHE_DIR)/'dataset.pkl').is_file():
    evalset = tokenizer.read_dataset((Path(EVAL_CACHE_DIR)/'dataset.pkl'))
else:
    evalset = tokenizer.tokenize_dataset(EVALSET, EVAL_CACHE_DIR, True, True)
trainset = PretrainSet(trainset, CONTEXT)
evalset = PretrainSet(evalset, CONTEXT)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
trainloader.len = MAX_ITERS*GRAD_ACC_STEPS
evalloader = DataLoader(evalset, batch_size=BATCH_SIZE, shuffle=False)
evalloader.len = EVAL_ITERS
print(f'Total training samples = {len(trainset)}')

# =============================================================
# Model, Optimizer, Criterion init and Checkpoint load
# =============================================================
net = GPT(
    EMB_DIM, HEADS, NUM_LAYERS,
    tokenizer.n_vocab, CONTEXT,
    DROPOUT, DROPOUT, DROPOUT,
)
optimizer = net.get_optimizer(lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
net_state, optim_state, itr, best, kwargs = load_checkpoint(LOAD)
criterion = nn.CrossEntropyLoss()
if net_state is not None:
    net.load_state_dict(net_state)
    optimizer.load_state_dict(optim_state)
net.to(DEVICE)
print(f'Total model parameters = {net.n_params} = {net.n_params/1e6}M')

# =============================================================
# Learning Rate Decay Scheduler (cosine with warmup)
# =============================================================
def get_lr(iter):
    """Refs:
            https://github.com/karpathy/nanoGPT/blob/master/train.py
    """
    # 1) linear warmup for warmup_iters steps
    if iter < WARMUP_ITERS:
        return LR * iter / WARMUP_ITERS
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (LR - MIN_LR)

# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
trainloss, evalloss, log_trainloss = 0, 0, 0
trainloader, evalloader = iter(trainloader), iter(evalloader)
net.train()
for itr in range(itr, MAX_ITERS):
    # cosine scheduler with warmup learning rate decay
    if DECAY_LR:
        lr = get_lr(itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    optimizer.zero_grad()

    # forward, loss, backward with grad. accumulation
    loss_ = 0
    for step in range(GRAD_ACC_STEPS):
        inp, tar = next(trainloader)
        inp, tar = inp.to(DEVICE), tar.to(DEVICE)

        logits = net(inp)
        loss = criterion(logits.reshape(-1, tokenizer.n_vocab), tar.reshape(-1))
        loss.backward()

        loss_ += loss.item()

    # optimize params
    loss_ = loss_/GRAD_ACC_STEPS
    trainloss += loss_
    log_trainloss += loss_
    if GRADIENT_CLIP is not None:
        nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
    optimizer.step()

    # print info.
    if itr%PRINT_INTERVAL == 0:
        log_data = f"iteration: {itr}/{MAX_ITERS}, \ttrain loss: {log_trainloss/PRINT_INTERVAL}"
        print(log_data)
        log_trainloss = 0

    # if itr%EVAL_INTERVAL == 0:
    #     net.eval()
    #     trainloss = trainloss/EVAL_INTERVAL
    #     if trainloss < best:
    #         best = trainloss
    #         save_checkpoint(
    #             in_int2tk, out_int2tk, '<S>', '<E>', '<P>', '<U>',
    #             net, epoch, trainloss, best, LOGDIR/f'best.pt',
    #         )
    #     save_checkpoint(
    #         in_int2tk, out_int2tk, '<S>', '<E>', '<P>', '<U>',
    #         net, epoch, trainloss, best, LOGDIR/f'checkpoint.pt',
    #     )
    #     write_pred(inp[0], pred[0], in_int2tk, out_int2tk, LOGDIR/'predictions.txt', label=f'epoch = {epoch}')
    #     logfile = LOGDIR/'log.txt'
    #     log_data = f"epoch: {epoch}/{EPOCHS}, \tloss: {trainloss},\t best_loss: {best}"
    #     print(f'{"-"*100}\n{log_data}\n{"-"*100}')
    #     with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
    #         fp.write(log_data + '\n')
    #     trainloss = 0
    #     net.train()

# =============================================================
# END
# =============================================================