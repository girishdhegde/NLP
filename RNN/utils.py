import numpy as np
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def save_checkpoint(hidden_size, num_layers, vocab_size, int2char, state_dict, epoch, loss, best, filename):
    checkpoint = {
        'HIDDEN_SIZE': hidden_size,
        'NUM_LAYERS': num_layers,
        'tokens': vocab_size,
        'state_dict': state_dict,
        'epoch': epoch,
        'loss': loss,
        'best': best,
        'int2char': int2char,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, net, device='cpu'):
    best = float('inf')
    int2char = None
    epoch = 0
    if filename is not None:
        if filename.is_file():
            load = torch.load(filename, map_location=device)
            net.load_state_dict(load['state_dict'])
            print('Checkpoint loaded successfully ...')
            if 'best' in load:
                best = load['best']
            if 'int2char' in load:
                int2char = load['int2char']
            if 'epoch' in load:
                epoch = load['epoch']
    return net, best, int2char, epoch


@torch.no_grad()
def predict(net, char, int2char, char2int, embeddings, hidden=None, top_k=1, device='cpu'):
    """ Function to get next character.
        author: girish d. hegde

    Refs:
        https://www.kaggle.com/code/ashukr/char-rnn/notebook

    Args:
        net (torch.nn.Module): trained model.
        char (str): Any character out of vocabulary.
        int2char (dict[int:str]): integer to character/token lookup.
        char2int (dict[int:str]): character/token to integer lookup.
        embeddings (torch.tensor): [vocab_size, vocab_size] - embeddings. vocab_size = len(int2char).
        hidden (torch.tensor): [num_layers, 1, hidden_size] - hidden state.
        top_k (int): topk sampling.
        device (torch.device): cpu or cuda.

    Returns:
        str: output character.
        torch.tensor: updated hidden state of shape [num_layers, 1, hidden_size].
    """
    enc = char2int[char]
    emb = embeddings[enc]
    inp = emb[None, None, :]  # shape = [timesteps = 1, bs = 1, vocabsize]
    # get prediction from nn
    hidden = net.init_hidden(1, inp.device) if hidden is None else hidden
    pred, hidden = net(inp, hidden)
    pred = F.softmax(pred[:, 0, :], -1).data.cpu()
    # get top characters
    pred, top_ch = pred.topk(top_k)
    top_ch = top_ch.numpy().squeeze()
    # select the likely next character with some element of randomness
    pred = pred.numpy().squeeze()
    char = np.random.choice(top_ch, p=pred/pred.sum()) if top_k > 1 else int(top_ch)
    return int2char[char], hidden


@torch.no_grad()
def sample(net, int2char, top_k=1, prime=None, max_size=12, device='cpu', eow='<E>'):
    """ Function to sample words from trained model.
        author: girish d. hegde

    Refs:
        https://www.kaggle.com/code/ashukr/char-rnn/notebook

    Args:
        net (torch.nn.Module): trained model.
        int2char (dict[int:str]): integer to character/token lookup.
        top_k (int): topk sampling.
        prime (str): Any initial string(only characters belonging to vocabulary are supported).
        max_size (int): Max output string size.
        device (torch.device): cpu or cuda.
        eow (str): end of word character.

    Returns:
        str: output string.
    """
    char2int = {ch: i for i, ch in int2char.items()}
    embeddings = torch.eye(len(int2char)).float().to(device)

    net = net.to(device)
    net.eval()

    # Run through the prime/initial characters
    prime = ['<S>', ] + (list(prime.lower()) if prime is not None else [])
    chars = prime[1:]
    hdn = net.init_hidden(1, device)
    for ch in prime:
        char, hdn = predict(net, ch, int2char, char2int, embeddings, hdn, top_k, device)
    chars.append(char)

    # Now pass in the previous character and get a new one
    while (char != eow) and (len(chars) <= max_size):
        char, hdn = predict(net, chars[-1], int2char, char2int, embeddings, hdn, top_k, device)
        chars.append(char)
    chars = chars if chars[-1] != eow else chars[:-1]

    return ''.join(chars)