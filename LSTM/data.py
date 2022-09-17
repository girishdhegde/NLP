import re
import random
from pathlib import Path
from copy import deepcopy
import json

import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class WordTokenizer:
    """ word level tokenization
        author: giris d. hegde

    Refs:
        https://towardsdatascience.com/dynamic-word-tokenization-with-regex-tokenizer-801ae839d1cd
    """
    def __init__(self, ): pass

    @classmethod
    def read(cls, filename, encoding='utf-8'):
        """ Fuction to read file containing corpus of text.
            author: girish d. hegde

        Args:
            filename (Path): path to .txt file.
            encodeing (str): file encoding.

        Returns:
            str: corpus of text
        """
        with open(filename, encoding=encoding) as fp:
            corpus = fp.read()
        return corpus

    @classmethod
    def tokenize(cls, corpus, lowercase=True):
        """ Fucntion to tokenize corpus of text to tokens/words.
            author: girish d. hegde

        Args:
            corpus (str): text corpus of words.
            lowercase (bool): convert all tokens to lowercase.

        Returns:
            str: corpus with special characters space padded.
            list[str]: list of tokens/words of corpus.
            list[str]: unique tokens.
        """
        corpus = corpus.lower() if lowercase else corpus
        sp_chars = re.sub('[A-Za-z]', '', deepcopy(corpus)).replace(' ', '')
        sp_chars = set(sp_chars)
        for sp in sp_chars:
            corpus = corpus.replace(sp, f' {sp} ')
        words = corpus.split()
        tokens = list(set(words))
        return corpus, words, tokens

    @classmethod
    def filter_sparse(cls, words, tokens, frequency=1):
        """ Function to remove tokens with occarances less than given frequency.
            author: girish d. hegde

        Args:
            words (list[str]): list of tokens/words of corpus.
            tokens (list[str]): unique tokens/words.
            frequency (int): minimum token/word occarance threshold.

        Returns:
            list[str]: words - list of filtered tokens/words.
            list[str]: tokens - remaining tokens after filtering.
            dict[str:int]: token2freq - dictionary/lookup of remaining token frequencies.
        """
        token_count = [words.count(token) for token in tokens]
        token2freq = dict(zip(tokens, token_count))
        words = [token for token in words if token2freq[token] >= frequency]
        tokens = [token for token in tokens if token2freq[token] >= frequency]
        token2freq = {token:token2freq[token] for token in tokens}
        return words, tokens, token2freq

    @classmethod
    def get_lookups(cls, tokens):
        """ Fucntion to get tokens to integer and interger to token lookups.
            author: girish d. hegde

        Args:
             tokens (list[str]): unique tokens/words.

        Returns:
            dict[int:str]: int2token - integer to token lookup.
            dict[int:str]: token2int - token to integer lookup.
       """
        int2token = dict(enumerate(tokens))
        token2int = {tk:i for i, tk in int2token.items()}
        return int2token, token2int

    @classmethod
    def write_json(cls, filename, int2token):
        """ Functiont to write integer to tokens dictionary into json file.
            author: girish d. hegde

        Args:
            filename (Path): path to .json file.
            int2token (dict[int:str]): integer to token lookup.
        """
        with open(filename, 'w') as fp:
            json.dump(int2token, fp, indent=4)
        return True

    @classmethod
    def read_json(cls, filename):
        """ Functiont to read integer to tokens json file.
            author: girish d. hegde

        Args:
            filename (Path): path to .json file.

        Returns:
            dict[int:str]: int2token - integer to token lookup.
        """

        with open(filename) as fp:
            int2token = json.load(fp)
        int2token = {int(k):v for k, v in int2token.items()}
        return int2token

    @classmethod
    def run(cls, filename, lowercase=True, min_frequency=1, out_json=None, encoding='utf-8', verbose=False):
        """ Function run tokenization pipeline on text corpus file.
                1. read file into string corpus.
                2. convert corpus into tokens.
                3. filter less common tokens.
                4. get tokens to integer and vice versa lookups.
                5. write token lookups.
            author: girish d. hegde

        Args:
            filename (Path): path to .txt file containg corpus of text data.
            lowercase (bool): convert all tokens to lowercase.
            min_frequency (int): minimum token/word occarance threshold.
            out_json (Path): path to .json file for writing int2token lookup.
            encodeing (str): input file encoding.
            verbose (bool): print info.

        Returns:
            list[str]: words - list of tokens/words in corpus.
            list[str]: tokens - unique tokens.
            dict[str:int]: token2ifreq - dictionary/lookup of token occarance frequencies.
            dict[int:str]: int2token - integer to token lookup.
            dict[int:str]: token2int - token to integer lookup.
        """
        corpus = cls.read(filename, encoding)
        if verbose: print('Total characters in corpus = ', len(corpus))
        corpus, words, tokens = cls.tokenize(corpus, lowercase)
        if verbose: print('Total unique tokens = ', len(tokens))
        if verbose: print('Total words/tokens in corpus = ', len(words))
        words, tokens, token2freq = cls.filter_sparse(words, tokens, min_frequency)
        if verbose: print('Total unique tokens after filtering by count = ', len(tokens))
        if verbose: print('Total words/tokens in corpus after filtering by count = ', len(words))
        int2token, token2int = cls.get_lookups(tokens)
        if out_json is not None: cls.write_json(out_json, int2token)
        return words, tokens, token2freq, int2token, token2int


class TextSet(Dataset):
    """ Pytorch word level Dataset class for text corpus.
        author: girish d. hegde

    Args:
        filename (str): .txt filepath.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
    """
    def __init__(self, filename, sequence_length=50, device='cuda'):
        super().__init__()
        with open(filename, encoding='utf-8') as fp:
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
