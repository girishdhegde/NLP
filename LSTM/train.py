from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lstm import WordLSTM
from data import WordTokenizer, TextSet
from utils import save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"

EMBEDDING_DIM = 32
HIDDEN_SIZE = 256
NUM_LAYERS = 3
SEQ_LEN = 25

LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 200
GRADIENT_CLIP = None  # 5

LOGDIR = Path('./data/runs')
PRINT_FREQ = 100  # print info. frequency wrt iterations.
SAVE_FREQ = 1  # checkpoint and log save frequency wrt epochs.

DATAPATH = './data/datasets/corpus.txt'
LOAD = Path('./data/runs/best.pt')  # or None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)

words, tokens, token2freq, int2token, token2int = WordTokenizer.run(
    DATAPATH, lowercase=True, min_frequency=10,
    exclude=(), out_json='./data/runs/tokens.json',
    encoding='utf-8', verbose=True
)

net = WordLSTM(len(int2token), EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0)
net, best, int2token_, start_epoch = load_checkpoint(LOAD, net, DEVICE)
int2token = int2token if int2token_ is None else int2token_
textset = TextSet(words, int2token, SEQ_LEN, DEVICE)
net.to(DEVICE)
params = sum(p.numel() for p in net.parameters())

iterations = (len(textset)//BATCH_SIZE) + 1
print('Total training samples = ', len(textset))
print(f'Total model parameters = {params} = {params/1e6}M')
# print(net)

trainloader = DataLoader(textset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


net.train()
for epoch in range(start_epoch, EPOCHS):
    trainloss = 0
    for iteration, (inp, target) in enumerate(trainloader):
        inp, target = inp.permute(1, 0), target.permute(1, 0)  # (bs, seqlen) -> (seqlen, bs)
        timesteps, bs = inp.shape
        # h_t, c_t = net.init_hidden(bs, DEVICE)
        h_t = torch.zeros(NUM_LAYERS, bs, HIDDEN_SIZE, device=DEVICE)
        c_t = torch.zeros(NUM_LAYERS, bs, HIDDEN_SIZE, device=DEVICE)

        optimizer.zero_grad()
        pred, states = net(inp, (h_t, c_t))

        loss = criterion(pred.reshape(-1, textset.vocab_size), target.reshape(-1))
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
            net.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, int2token,
            net.state_dict(), epoch, trainloss, best, LOGDIR/f'best.pt'
        )

    if epoch%SAVE_FREQ == 0:
        save_checkpoint(
            net.vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, int2token,
            net.state_dict(), epoch, trainloss, best, LOGDIR/f'checkpoint.pt'
        )
        write_pred(pred[:, 0, :], int2token, LOGDIR/'predictions.txt', label=f'epoch = {epoch}')
        logfile = LOGDIR/'log.txt'
        log_data = f"epoch: {epoch}/{EPOCHS}, \tloss: {trainloss},\t best_loss: {best}"
        print(f'{"-"*100}\n{log_data}\n{"-"*100}')
        with open(logfile, 'a' if logfile.is_file() else 'w') as fp:
            fp.write(log_data + '\n')