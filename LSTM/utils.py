"""
Refs:
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import WordTokenizer


__author__ = "__Girish_Hegde__"


def save_checkpoint(vocab_size, embedding_dim, hidden_size, num_layers, int2token, state_dict, epoch, loss, best, filename):
    checkpoint = {
        'VOCAB_SIZE': vocab_size,
        'EMBEDDING_DIM': embedding_dim,
        'HIDDEN_SIZE': hidden_size,
        'NUM_LAYERS': num_layers,
        'int2token': int2token,
        'state_dict': state_dict,
        'epoch': epoch,
        'loss': loss,
        'best': best,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, net, device='cpu'):
    best = float('inf')
    int2token = None
    epoch = 0
    if filename is not None:
        if filename.is_file():
            load = torch.load(filename, map_location=device)
            net.load_state_dict(load['state_dict'])
            print('Checkpoint loaded successfully ...')
            if 'best' in load:
                best = load['best']
            if 'int2token' in load:
                int2token = load['int2token']
            if 'epoch' in load:
                epoch = load['epoch']
    return net, best, int2token, epoch


@torch.no_grad()
def logits2text(logits, int2token, ):
    """ Function to convert model prediction logits into words.
        author: girish d. hegde

    Args:
        logits (torch.tensor[float]): [seq_len, vocab_size] - logits.
        int2token (dict[int:str]): [vocab_size, ] - int to token lookup.

    Returns:
        str: output string(set of words).
    """
    positions = torch.argmax(logits.detach().cpu(), dim=-1)
    words = [int2token[p.item()] for p in positions]
    return ' '.join(words)


def write_pred(logits, int2token, filename, label=''):
    text = logits2text(logits, int2token)
    data = f'\n{"-"*100}\n{label}\n{"-"*100}\n{text}\n{"-"*100}'
    with open(filename,'a' if Path(filename).is_file() else 'w') as fp:
         fp.write(data)
    return text


@torch.no_grad()
def predict(net, token, int2token, token2int, init_states=None, top_k=1, device='cpu'):
    """ Function to get next token.
        author: girish d. hegde

    Args:
        net (torch.nn.Module): trained model.
        token (str): Any token out of vocabulary.
        int2token (dict[int:str]): integer to token lookup.
        token2int (dict[int:str]): token to integer lookup.
        init_states (tuple[torch.tensor]): (hidden_state, cell_state)
                                                [num_layers, 1, hidden_size] - previous hidden state.
                                                [num_layers, 1, hidden_size] - previous cell state.
        top_k (int): topk sampling.
        device (torch.device): cpu or cuda.

    Returns:
        str: output token.
        tuple: (updated hidden, updated cell state)
            torch.tensor: hidden state - [num_layers, 1, hidden_size].
            torch.tensor: cell state - [num_layers, 1, hidden_size].
    """
    enc = torch.tensor([[token2int[token.lower()]]], dtype=torch.int64, device=device)
    # get prediction from nn
    init_states = net.init_hidden(1, device) if init_states is None else init_states
    pred, init_states = net(enc, init_states)
    pred = F.softmax(pred[0, :, :], -1).data.cpu()
    # get top characters
    pred, top_tk = pred.topk(top_k)
    top_tk = top_tk.numpy().squeeze()
    # select the likely next token with some element of randomness
    pred = pred.numpy().squeeze()
    tk = np.random.choice(top_tk, p=pred/pred.sum()) if top_k > 1 else int(top_tk)
    return int2token[tk], init_states


@torch.no_grad()
def sample(net, int2token, top_k=1, prime='The', max_size=100, device='cpu'):
    """ Function to sample text from trained model.
        author: girish d. hegde

    Args:
        net (torch.nn.Module): trained model.
        int2token (dict[int:str]): integer to token lookup.
        top_k (int): topk sampling.
        prime (str): Any initial space seperated string(only tokens belonging to vocabulary are supported).
        max_size (int): Max output tokens/words size.
        device (torch.device): cpu or cuda.

    Returns:
        str: output string/text.
    """
    token2int = {tk: i for i, tk in int2token.items()}

    net = net.to(device)
    net.eval()

    # Run through the prime/initial characters
    _, prime, _ = WordTokenizer.tokenize(prime, lowercase=True)
    prime = [w for w in prime if w in token2int]
    output = prime[:]
    init_states = net.init_hidden(1, device)
    for tk in prime:
        tk, init_states = predict(net, tk, int2token, token2int, init_states, top_k, device)
    output.append(tk)

    # Now pass in the previous token and get a new one
    while len(output) <= max_size:
        tk, init_states = predict(net, output[-1], int2token, token2int, init_states, top_k, device)
        output.append(tk)

    return ' '.join(output)

