import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rnn import CharRNN
from data import WordSet, same_timesteps_collate_fn


__author__ = "__Girish_Hegde__"


HIDDEN_SIZE = 64
NUM_LAYERS = 3

LR = 2e-3
BATCH_SIZE = 16
EPOCHS = 100
GRADIENT_CLIP = 5

LOGDIR = Path('./data/runs')
LOG_FREQ = 10  # print/save frequency wrt iterations.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)


wordset = WordSet('./data/sanskrit_words.txt', DEVICE)
trainloader = DataLoader(wordset, batch_size=BATCH_SIZE, collate_fn=same_timesteps_collate_fn, shuffle=True)
iterations = (len(wordset)//BATCH_SIZE) + 1
print('Total data samples = ', len(wordset))
print(f'Vocab size = {wordset.vocab_size}')
print(f'Total characters = {sum(wordset.chars_per_word)}')


net = CharRNN(wordset.vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
params = sum(p.numel() for p in net.parameters())
print(f'Total model parameters = {params} = {params/1e6}M')
# print(net)


optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


net.train()

for epoch in range(EPOCHS):
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

        if iteration % LOG_FREQ == 0:
            print(f"epoch: {epoch + 1}/{EPOCHS},\t",
                f"iteration: {iteration}/{iterations},\t",
                f"loss: {loss.item()}")