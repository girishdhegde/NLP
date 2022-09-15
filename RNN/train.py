from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rnn import CharRNN
from data import Word2ClassSet, Word2WordSet, word2word_collate_fn, word2class_collate_fn
from utils import save_checkpoint, load_checkpoint


__author__ = "__Girish_Hegde__"


HIDDEN_SIZE = 256
NUM_LAYERS = 3

LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 1000
GRADIENT_CLIP = None  # 5

LOGDIR = Path('./data/runs')
PRINT_FREQ = 100  # print info. frequency wrt iterations.
SAVE_FREQ = 10  # checkpoint and log save frequency wrt epochs.

LOAD = Path('./data/runs/best.pt')  # or None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)


# textset = Word2ClassSet('../data/names', DEVICE)
textset = Word2WordSet('../data/names.txt', DEVICE)
iterations = (len(textset)//BATCH_SIZE) + 1
print('Total training samples = ', len(textset))
print(f'Vocab size = {textset.vocab_size}')

net = CharRNN(textset.vocab_size, HIDDEN_SIZE, NUM_LAYERS, dropout=0)
net, best, int2char, start_epoch = load_checkpoint(LOAD, net, DEVICE)
textset = Word2WordSet('../data/names.txt', DEVICE, int2char) if int2char is not None else textset
net.to(DEVICE)
params = sum(p.numel() for p in net.parameters())
print(f'Total model parameters = {params} = {params/1e6}M')
# print(net)

trainloader = DataLoader(textset, batch_size=BATCH_SIZE, collate_fn=word2word_collate_fn, shuffle=True)
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


net.train()
for epoch in range(start_epoch, EPOCHS):
    trainloss = 0
    for iteration, (inp, target, nchars) in enumerate(trainloader):
        inp, target = inp.permute(1, 0, 2), target.permute(1, 0, 2)
        timesteps, bs, inpsize = inp.shape
        # hdn = net.rnn.init_hidden(bs, DEVICE)
        hdn = torch.zeros(NUM_LAYERS, bs, HIDDEN_SIZE, device=DEVICE)

        optimizer.zero_grad()
        pred, hdn = net(inp, hdn)
        # pred = pred.permute(1, 0, 2)

        loss = criterion(pred.reshape(-1, inpsize), target.reshape(-1, inpsize))
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
        save_checkpoint(
            HIDDEN_SIZE, NUM_LAYERS, textset.vocab_size, textset.int2char,
            net.state_dict(), epoch, trainloss, best, LOGDIR/f'best.pt'
        )

    if epoch%SAVE_FREQ == 0:
        save_checkpoint(
            HIDDEN_SIZE, NUM_LAYERS, textset.vocab_size, textset.int2char,
            net.state_dict(), epoch, trainloss, best, LOGDIR/f'checkpoint.pt'
        )
        logfile = LOGDIR/'log.txt'
        log_data = f"epoch: {epoch}/{EPOCHS}, \tloss: {trainloss},\t best_loss: {best}"
        print(f'{"-"*100}\n{log_data}\n{"-"*100}')
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')