import re

import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class WordSet(Dataset):
    """ Pytorch Dataset class for set of word text documents.
        author: girish d. hegde

    Args:
        filename (str): .txt filepath.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
    """
    def __init__(self, filename, device='cuda'):
        super().__init__()
        with open(filename) as fp:
            text = fp.readlines()
        # preprocess/clean text data
        text = [re.sub('[^A-Za-z0-9]+', '', line) for line in text]
        text = [word.lower() for word in text]
        text = [word for word in text if (len(word) > 2) and (len(word) < 13)]
        # generate char to int dictionary/lookup tables
        allchars = ''.join(text)
        chars = list(set(allchars))
        chars.append('eow')  # end of word
        self.vocab_size = len(chars)
        self.int2char = dict(enumerate(chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        # generate one-hot char and word embeddings
        self.encoded = [[self.char2int[ch] for ch in word] + [self.vocab_size - 1, ] for word in text]
        self.embeddings = torch.eye(self.vocab_size).float().to(device)
        self.word_embeddings = [self.embeddings[enc] for enc in self.encoded]
        self.chars_per_word = [len(word) for word in text]

    def __len__(self):
        return len(self.word_embeddings)

    def __getitem__(self, index):
        """
        Returns:
            torch.tensor: [timesteps, vocab_size] - input char embeddings.
            torch.tensor: [timesteps, vocab_size] - label char embeddings.
            int: timesteps.

        Note:
            timesteps = len(word) - 1, - 1 because eow char is not included.
        """
        return self.word_embeddings[index][:-1], self.word_embeddings[index][1:], self.chars_per_word[index]


def same_timesteps_collate_fn(batch):
    """ Pytorch collate_fn to get pad all individual data in a batch to get same size.
        author: girish d. hegde

    Args:
        batch (tuple[torch.tensor]): [N, ...] - list of batch data with different timesteps.

    Returns:
        torch.tensor: [N, max(nchars), ...] - batch input data.
        torch.tensor: [N, max(nchars), ...] - batch label data.
    """
    bs, vocab_size = len(batch), batch[0][0].shape[-1]
    device, dtype = batch[0][0].device, batch[0][0].dtype
    nchars = [d[2] for d in batch]
    max_len = max(nchars)
    inputs_ = torch.zeros((bs, max_len, vocab_size), device=device, dtype=dtype)
    labels_ = torch.zeros((bs, max_len, vocab_size), device=device, dtype=dtype)
    for i, n in enumerate(nchars):
        inputs_[i, :n] = batch[i][0]
        labels_[i, :n] = batch[i][1]
    return inputs_, labels_, nchars


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    wordset = WordSet('./data/sanskrit_words.txt', 'cuda')
    print('Total data samples = ', len(wordset))
    print(f'Vocab size = {wordset.vocab_size}')
    print(f'Total characters = {sum(wordset.chars_per_word)}')
    loader = DataLoader(wordset, batch_size=4, collate_fn=same_timesteps_collate_fn, shuffle=True)
    loader = iter(loader)
    inputs, labels, nchars = next(loader)
    print(f'{inputs.shape = }, {labels.shape = }, {nchars = }')

    from rnn import RNN
    net = RNN(wordset.vocab_size, 512, 10).cuda()
    hdn = net.init_hidden(inputs.shape[0], inputs.device)
    out, hdn = net(inputs.permute(1, 0, 2), hdn)
    out.sum().backward()
    print(f'{out.shape = }')
    print(f'{hdn.shape = }')