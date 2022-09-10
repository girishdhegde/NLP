import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rnn import CharRNN
from data import WordSet, same_timesteps_collate_fn
from utils import save_checkpoint


__author__ = "__Girish_Hegde__"


HIDDEN_SIZE = 512
NUM_LAYERS = 3

LR = 2e-5
BATCH_SIZE = 32
EPOCHS = 1000
GRADIENT_CLIP = 5

LOGDIR = Path('./data/runs')
PRINT_FREQ = 10  # print info. frequency wrt iterations.
SAVE_FREQ = 10  # checkpoint and log save frequency wrt epochs.

LOAD = None
LOAD = Path('./data/runs/best.pt')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)


wordset = WordSet('./data/sanskrit_words.txt', DEVICE)
trainloader = DataLoader(wordset, batch_size=BATCH_SIZE, collate_fn=same_timesteps_collate_fn, shuffle=True)
iterations = (len(wordset)//BATCH_SIZE) + 1
print('Total data samples = ', len(wordset))
print(f'Vocab size = {wordset.vocab_size}')
print(f'Total characters = {sum(wordset.chars_per_word)}')


net = CharRNN(wordset.vocab_size, HIDDEN_SIZE, NUM_LAYERS, dropout=0).to(DEVICE)
params = sum(p.numel() for p in net.parameters())
print(f'Total model parameters = {params} = {params/1e6}M')
# print(net)

best = float('inf')
if LOAD is not None:
    if LOAD.is_file():
        load = torch.load(LOAD)
        net.load_state_dict(load['state_dict'])
        best = load['best']
        print('Model loaded ...')

optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


net.train()
for epoch in range(EPOCHS):
    trainloss = 0
    for iteration, (inp, target, nchars) in enumerate(trainloader):
        inp, target = inp.permute(1, 0, 2), target.permute(1, 0, 2)
        timesteps, bs, inpsize = inp.shape
        hdn = net.rnn.init_hidden(bs, DEVICE)

        optimizer.zero_grad()
        pred, hdn = net(inp, hdn)

        lbl = torch.argmax(target.reshape(-1, inpsize), dim=-1)
        loss = criterion(pred.reshape(-1, inpsize), lbl)
        loss.backward()
        if GRADIENT_CLIP is not None:
            nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
        optimizer.step()

        trainloss += loss.item()
        if iteration%PRINT_FREQ == 0:
            log_data = f"epoch: {epoch}/{EPOCHS},\titeration: {iteration}/{iterations},\tloss: {loss.item()},\t best_loss: {best}"
            print(log_data)

    trainloss = trainloss/iteration
    if trainloss < best:
        best = trainloss
        save_checkpoint(HIDDEN_SIZE, NUM_LAYERS, wordset.vocab_size, net.state_dict(), trainloss, best, LOGDIR/f'best.pt')

    if epoch%SAVE_FREQ == 0:
        save_checkpoint(HIDDEN_SIZE, NUM_LAYERS, wordset.vocab_size, net.state_dict(), trainloss, best, LOGDIR/f'checkpoint.pt')
        logfile = LOGDIR/'log.txt'
        log_data = f"epoch: {epoch}/{EPOCHS},\titeration: {iteration}/{iterations},\tloss: {trainloss},\t best_loss: {best}"
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')



# TODO: predict, topk predict