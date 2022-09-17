import re
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class TextSet(Dataset):
    """ Pytorch word level Dataset class for text corpus.
        author: girish d. hegde

    Args:
        filename (str): .txt filepath.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
    """
    def __init__(self, filename, sequence_length=50, device='cuda'):
        super().__init__()
        with open(filename) as fp:
            text = fp.read()
        text = text.replace('\n', '')
        # text = re.sub('[^A-Za-z0-9]+', '', text)
        text = text.lower()

        # generate char to int dictionary/lookup tables
        chars = list(set(text))
        self.sequence_length = sequence_length
        self.vocab_size = len(chars)
        self.int2char = dict(enumerate(chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        # generate one-hot char and word embeddings
        self.encoded = [self.char2int[ch] for ch in text]
        self.embeddings = torch.eye(self.vocab_size).float().to(device)
        self.char_embeddings = torch.stack([self.embeddings[enc] for enc in self.encoded])
        self.len = len(text) - (sequence_length + 1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Returns:
            torch.tensor: [timesteps, vocab_size] - input char embeddings.
            torch.tensor: [timesteps, vocab_size] - label char embeddings.
        """
        start = random.randint(0, self.len)
        end = start + self.sequence_length
        return self.char_embeddings[start:end], self.char_embeddings[start + 1:end + 1]
