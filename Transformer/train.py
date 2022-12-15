from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformer import Transformer
from data import InOutTokenizer, TranslationSet
# from utils import save_checkpoint, load_checkpoint, write_pred


__author__ = "__Girish_Hegde__"


EMB_DIM = 32
HEADS = 8
NUM_LAYERS = 3
PRE_ATTN_ACT = None
POST_ATTN_ACT = None
FFN_ATTN_ACT = nn.ReLU
DROPOUT = 0.0

LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 100
GRADIENT_CLIP = None  # 5

LOGDIR = Path('./data/runs')
PRINT_FREQ = 100  # print info. frequency wrt iterations.
SAVE_FREQ = 1  # checkpoint and log save frequency wrt epochs.

DATAPATH = './data/tasks_train_simple.txt'
LOAD = Path('./data/runs/best.pt')  # or None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOGDIR.mkdir(parents=True, exist_ok=True)


in_corpus, out_corpus, in_int2tk, out_int2tk = InOutTokenizer.run(
    DATAPATH, in_token='IN:', out_token='OUT:',
    start_token='<S>', end_token='<E>', lowercase=False,
    out_json=LOGDIR/'tokens.json', encoding='utf-8',
    verbose=True,
)


net = Transformer(
    EMB_DIM, len(in_int2tk), len(out_int2tk),
    HEADS, NUM_LAYERS, PRE_ATTN_ACT, POST_ATTN_ACT,
    FFN_ATTN_ACT, DROPOUT,
)
net, start_epoch, best, in_int2tk_, out_int2tk_, start_token, end_token = load_checkpoint(LOAD, net, DEVICE)
in_int2tk = in_int2tk if in_int2tk_ is None else in_int2tk_
out_int2tk = out_int2tk if out_int2tk_ is None else out_int2tk_
translation_set = TranslationSet(in_corpus, out_corpus, in_int2tk, out_int2tk, DEVICE)
InOutTokenizer.write_json(LOGDIR/'tokens.json', in_int2tk, out_int2tk)
net.to(DEVICE)


params = sum(p.numel() for p in net.parameters())
iterations = (len(translation_set)//BATCH_SIZE) + 1
print('Total training samples = ', len(translation_set))
print(f'Total model parameters = {params} = {params/1e6}M')
# print(net)

#!!!!!!!!!!!!!!!!!!!!!!!!!
# collate function is required
trainloader = DataLoader(translation_set, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


net.train()
for epoch in range(start_epoch, EPOCHS):
    trainloss = 0
    for iteration, (inp, target) in enumerate(trainloader):
        inp, target = inp.permute(1, 0), target.permute(1, 0)  # (bs, seqlen) -> (seqlen, bs)
        timesteps, bs = inp.shape
        h_t, c_t = net.init_hidden(bs, DEVICE)
        # h_t = torch.zeros(NUM_LAYERS, bs, HIDDEN_SIZE, device=DEVICE)
        # c_t = torch.zeros(NUM_LAYERS, bs, HIDDEN_SIZE, device=DEVICE)

        optimizer.zero_grad()
        pred, states = net(inp, (h_t, c_t))

        loss = criterion(pred.reshape(-1, translation_set.vocab_size), target.reshape(-1))
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