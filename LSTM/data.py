import re
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class Word2WordSet(Dataset):
    """ Pytorch Dataset class for set of word text documents.
        author: girish d. hegde

    Args:
        filename (str): .txt filepath.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
        int2char (dict[int:str]): precalculated int2char lookup.
    """
    def __init__(self, filename, device='cuda', int2char=None):
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
        chars += ['<S>', '<E>']  # start of word and end of word
        self.vocab_size = len(chars)
        self.int2char = dict(enumerate(chars)) if int2char is None else int2char
        self.char2int = {ch: i for i, ch in self.int2char.items()}
        # generate one-hot char and word embeddings
        sow, eow = self.char2int['<S>'], self.char2int['<E>']
        self.encoded = [[sow, ] + [self.char2int[ch] for ch in word] + [eow, ] for word in text]
        self.embeddings = torch.eye(self.vocab_size).float().to(device)
        self.word_embeddings = [self.embeddings[enc] for enc in self.encoded]
        self.chars_per_word = [len(word) + 1 for word in text]

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


class Word2ClassSet(Dataset):
    """ Pytorch Dataset class for word classification.
        author: girish d. hegde

    Args:
        dirname (str): path to Directory containing data.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
        int2char (dict[int:str]): precalculated int2char lookup.

    Note:
        Data directory structure:
            dirname/
                class_0.txt -> label = 0.
                class_1.txt -> label = 1.
                ...
                class_n.txt -> label = n.
    """
    def __init__(self, dirname, device='cuda', int2char=None):
        super().__init__()
        directory = Path(dirname)
        textfiles = list(directory.glob('**/*.txt'))
        self.classnames = [fp.stem for fp in textfiles]
        self.classes = []
        data = []
        # read data
        for i, filename in enumerate(textfiles):
            with open(filename, encoding="utf8") as fp:
                text = fp.read()
                # process/clean data
                text = re.sub('[^A-Za-z0-9\n]+', '', text)
                text = text.lower().split('\n')
                # get class ids.
                text = [t.strip() for t in text]
                text = [t for t in text if len(t)]
                self.classes += [i for _ in text]
                data += text
        text = data
        # generate char to int dictionary/lookup tables
        allchars = ''.join(text)
        chars = list(set(allchars))
        self.vocab_size = len(chars)
        self.int2char = dict(enumerate(chars)) if int2char is None else int2char
        self.char2int = {ch: i for i, ch in self.int2char.items()}
        # generate one-hot char and word embeddings
        self.encoded = [[self.char2int[ch] for ch in word] for word in text]
        self.embeddings = torch.eye(self.vocab_size).float().to(device)
        self.word_embeddings = [self.embeddings[enc] for enc in self.encoded]
        self.chars_per_word = [len(word) for word in text]
        self.classes = torch.tensor(self.classes).long().to(device)

    def __len__(self):
        return len(self.word_embeddings)

    def __getitem__(self, index):
        """
        Returns:
            torch.tensor[float]: [timesteps, vocab_size] - input char embeddings.
            torch.tensor[long]:  label/class id.
            int: timesteps.

        Note:
            timesteps = len(word).
        """
        return self.word_embeddings[index], self.classes[index], self.chars_per_word[index]


def word2word_collate_fn(batch):
    """ Pytorch collate_fn to get pad all individual data in a batch to get same size.
        author: girish d. hegde

    Args:
        batch (tuple[torch.tensor]): [N, ...] - list of batch data with different timesteps.

    Returns:
        torch.tensor: [N, max(nchars), ...] - batch input data.
        torch.tensor: [N, max(nchars), ...] - batch label data.
        torch.tensor: [N, ] - batch total characters data.
    """
    bs, vocab_size = len(batch), batch[0][0].shape[-1]
    device, dtype = batch[0][0].device, batch[0][0].dtype
    nchars = [d[2] for d in batch]
    max_len = max(nchars)
    inputs_ = torch.zeros((bs, max_len, vocab_size), device=device, dtype=dtype)
    labels_ = torch.zeros((bs, max_len, vocab_size), device=device, dtype=batch[0][1].dtype)
    for i, n in enumerate(nchars):
        inputs_[i, :n] = batch[i][0]
        labels_[i, :n] = batch[i][1]
    return inputs_, labels_, nchars


def word2class_collate_fn(batch):
    """ Pytorch collate_fn to get pad all individual data in a batch to get same size.
        author: girish d. hegde

    Args:
        batch (tuple[torch.tensor]): [N, ...] - list of batch data with different timesteps.

    Returns:
        torch.tensor: [N, max(nchars), ...] - batch input data.
        torch.tensor: [N, ] - batch label data.
        torch.tensor: [N, ] - batch total characters data.
    """
    bs, vocab_size = len(batch), batch[0][0].shape[-1]
    device, dtype = batch[0][0].device, batch[0][0].dtype
    nchars = [d[2] for d in batch]
    labels_ = torch.stack([d[1] for d in batch])
    max_len = max(nchars)
    inputs_ = torch.zeros((bs, max_len, vocab_size), device=device, dtype=dtype)
    for i, n in enumerate(nchars):
        inputs_[i, :n] = batch[i][0]
    return inputs_, labels_, nchars