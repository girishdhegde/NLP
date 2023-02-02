import math
import time
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gpt import GPT
from data import BPETokenizer, CodeSet, SeqCollater
from utils import set_seed, save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"


# config file - (overrides the parameters given here)
CFG = './config/finetune.py'  # 'path/to/config/file.py'
# =============================================================
# Parameters
# (params inspired from from https://github.com/karpathy/nanoGPT/blob/master/train.py)
# =============================================================
# model
EMB_DIM = 256
HEADS = 8
NUM_LAYERS = 12
CONTEXT = 512
DROPOUT = 0.1
# logging
LOGDIR = Path('./data/code_finetune')
LOAD = LOGDIR/'ckpt.pt'  # or None
PRINT_INTERVAL = 10
# dataset
DATASET = './data/cache/codeparrot_train/dataset.pkl'
EVALSET = './data/cache/codeparrot_test/dataset.pkl'
N_TASKS = 10
# training
BATCH_SIZE = 2
GRAD_ACC_STEPS = 8  # used to simulate larger batch sizes
MAX_EPOCHS = 3
# EVAL_INTERVAL = 2000
EVAL_INTERVAL = 500
SAVE_EVERY = False  # save unique checkpoint at every eval interval.
GRADIENT_CLIP = None  # 5
# adamw optimizer
LR = 1e-5  # max learning rate
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95
# system
# dtype = 'bfloat16' # 'float32' or 'bfloat16'
# compile = True # use PyTorch 2.0 to compile the model to be faster
# init

# warning!!! executes codes in config file directly with no safety!
with open(CFG, 'r') as fp: exec(fp.read())  # import cfg settings
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)
set_seed(108)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cudnn.benchmark = True  # optimize backend algorithms
extras = {'n_tasks':N_TASKS, 'pre-training':False, }

# =============================================================
# Tokenizer, Dataset, Dataloader init
# =============================================================
tokenizer = BPETokenizer(n_tasks=N_TASKS)
trainset = tokenizer.read_dataset(DATASET)
code_token,  end_token, ignore_index = tokenizer.n_vocab - N_TASKS, tokenizer.n_vocab - 1, -1
extras['code_token'] = code_token
extras['end_token'] = end_token
trainset = CodeSet(trainset, tokenizer, code_token, end_token, ignore_index)
trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=SeqCollater(end_token, end_token)
)
print(f'Total training samples = {len(trainset)}')

if EVALSET is not None:
    evalset = tokenizer.read_dataset(EVALSET)
    evalset = CodeSet(evalset, tokenizer, code_token, end_token, ignore_index)
    evalloader = DataLoader(
        evalset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=SeqCollater(end_token, end_token)
    )
else:
    evalloader = None

# =============================================================
# Model, Optimizer, Criterion init and Checkpoint load
# =============================================================
net = GPT(
    EMB_DIM, HEADS, NUM_LAYERS,
    tokenizer.n_vocab, CONTEXT,
    DROPOUT, DROPOUT, DROPOUT,
)
net_state, optim_state, itr, best, kwargs = load_checkpoint(LOAD)
if net_state is not None:
    net.load_state_dict(net_state)
net.to(DEVICE)
optimizer = net.get_optimizer(lr=LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
if kwargs['pre-training']: # if pre-trainig weights are loaded reinit iteration to 1.
    itr, best, epoch = 1, float('inf'), 1
else:
    epoch = kwargs['epoch']
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
print(f'Total model parameters = {net.n_params} = {net.n_params/1e6}M')

# =============================================================
# Training loop - forward, backward, loss, optimize
# =============================================================
trainloss, valloss, log_trainloss, loss_ = 0, 0, 0, 0
net.train()
optimizer.zero_grad(set_to_none=True)
# set_to_none -> instead of filling grad with zero tensor set it to None
# reduce memory consumptionn + increases speed
print('Training ...')
start_time = time.perf_counter()
for epoch in range(epoch, MAX_EPOCHS + 1):
    for itr_, (inp, tar) in enumerate(trainloader):
        # =============================================================
        # Training
        # =============================================================
        # forward, loss, backward with grad. accumulation
        inp, tar = inp.to(DEVICE), tar.to(DEVICE)
        logits = net(inp)
        loss = criterion(logits.reshape(-1, tokenizer.n_vocab), tar.reshape(-1))
        loss.backward()
        loss_ += loss.item()

        # optimize params
        if itr_%(GRAD_ACC_STEPS) == 0:
            loss_ = loss_/GRAD_ACC_STEPS
            trainloss += loss_
            log_trainloss += loss_
            loss_ = 0
            if GRADIENT_CLIP is not None:
                nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # print info.
            if itr%PRINT_INTERVAL == 0:
                log_data = f"epoch: {epoch}/{MAX_EPOCHS}, \titeration: {itr}, \ttrain loss: {log_trainloss/PRINT_INTERVAL}"
                print(log_data)
                log_trainloss = 0

            # =============================================================
            # Validation
            # =============================================================
            if (itr%EVAL_INTERVAL == 0):
                print('Evaluating ...')
                trainloss = trainloss/EVAL_INTERVAL
                if evalloader is not None:
                    net.eval()
                    valloss = 0
                    with torch.no_grad():
                        for inp, tar in tqdm(evalloader, total=len(evalloader)):
                            inp, tar = inp.to(DEVICE), tar.to(DEVICE)
                            logits = net(inp)
                            loss = criterion(logits.reshape(-1, tokenizer.n_vocab), tar.reshape(-1))
                            valloss += loss.item()
                    net.train()

                    valloss = valloss/len(evalloader)
                else:
                    valloss = trainloss

                # =============================================================
                # Saving and Logging
                # =============================================================
                print('Saving checkpoint ...')
                ckpt_name = LOGDIR/'ckpt.pt' if not SAVE_EVERY else LOGDIR/f'ckpt_{itr}.pt'
                extras['epoch'] = epoch
                save_checkpoint(
                    net, optimizer, itr, valloss, trainloss, best, ckpt_name, **extras,
                )

                if valloss < best:
                    best = valloss
                    save_checkpoint(
                        net, optimizer, itr, valloss, trainloss, best, LOGDIR/'best.pt', **extras,
                    )

                valid_logits = logits[0][tar[0] != ignore_index]
                write_pred(inp[0], valid_logits, tokenizer, LOGDIR/'predictions.txt', label=f'iteration = {itr}')

                logfile = LOGDIR/'log.txt'
                log_data = f"epoch: {epoch}/{MAX_EPOCHS}, \titeration: {itr}, \tval loss: {valloss}, \ttrain loss: {trainloss}, \tbest loss: {best}"
                with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
                    fp.write(log_data + '\n')
                end_time = time.perf_counter()
                log_data = f'{log_data}, \t time: {(end_time - start_time)/60}M'
                print(f'{"-"*150}\n{log_data}\n{"-"*150}')

                trainloss = 0
                start_time = time.perf_counter()
                print('Training ...')
            itr += 1

# =============================================================
# END
# =============================================================