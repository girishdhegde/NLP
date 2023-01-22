import re
import random
from pathlib import Path
from copy import deepcopy
import pickle

import numpy as np
import tiktoken  # openai gpt2 bpe tokenizer
import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class TiktokenTokenizer:
    """ open-ai gpt2 bpe tokenizer based custom Text corpus tokenizer.
        author: girish d. hegde

        text dataset pickle file structure:
            dataset.pkl
                [
                    "paragraph 1 - string of words",
                    "paragraph 2",
                    ...
                    "paragraph n",
                ]

    Refs:
        https://github.com/openai/tiktoken
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.max_token_value

    @classmethod
    def read_dataset(cls, filename):
        """ Fuction to read pickle file containing dataset.
            author: girish d. hegde

        Args:
            filename (str/Path): path to dataset pickel file.

        Returns:
            list[str]/list[int]: dataset - dataset of string text or encoded dataset.
        """
        with open(filename, 'rb') as fp:
            dataset = pickle.load(fp)
        return dataset

    def tokenize_dataset(self, dataset, cache_dir=None, to_torch=True, verbose=False):
        """ Fucntion to tokenize dataset of text to token.
            author: girish d. hegde

        Args:
            dataset (list[str]): list of strings/text.
            cache_dir (str/Path): directory path to write tokenized dataset pickle file.
            to_torch (bool): convert encoding into torch.LongTensor.
            verbose (bool): print progress/info.

        Returns:
            list[list[int]]: tokenized_dataset - [list of encoded tokens for sample in dataset].
        """
        if isinstance(dataset, (str, Path)):
            dataset = self.read_dataset(dataset)

        tokenized_dataset = []
        ntokens = 0
        for example in dataset:
            if example:
                tokens = self.tokenizer.encode(example)
                tokens = torch.tensor(tokens, dtype=torch.int64) if to_torch else tokens
                tokenized_dataset.append(tokens)
                ntokens += len(tokens)

        if verbose: print(f'vocab size = {self.vocab_size}')
        if verbose: print(f'total tokens in dataset = {ntokens/1e6}M')

        if cache_dir is not None:
            if verbose: print('writing cache ...')
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True, parents=True)
            with open(cache_dir/'dataset.pkl', 'wb') as fp: pickle.dump(tokenized_dataset, fp)
            if verbose: print('caching done.')

        return tokenized_dataset

    def encode(self, text):
        """ Function to encode text into tokens.
            author: girish d. hegde

        Args:
            text (str): input text string.

        Returns:
            np.ndarray[int]: tokens - list encoded text.
        """
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, tokens):
        """ Function to decode tokens into text/string.
            author: girish d. hegde

        Args:
            tokens (list[int]): list of enocded tokens.

        Returns:
            str: text - decoded text/string.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        text = self.tokenizer.decode(tokens)
        return text


# TODO:
# add special task tokens
# prepare wikitext.py
# prepare codeparrot.py
# pretrain dataloader
# finetune dataloader
# collate functions