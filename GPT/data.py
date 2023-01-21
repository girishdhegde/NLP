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

    Args:
        cache_dir (str/Path): directory to/from write/read cache.
        dataset (str/Path): path to dataset pickle file.
        verbose (bool): print processing progress/info.

    Refs:
        https://github.com/openai/tiktoken
    """
    def __init__(self, cache_dir, dataset=None, verbose=False):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        if dataset is not None:
            self.run(cache_dir, dataset, verbose)
        else:
            self.read_cache(cache_dir, load_dataset=False)

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

    def tokenize(self, dataset, verbose=False):
        """ Fucntion to tokenize dataset of text to tokens and to update token lookups.
            author: girish d. hegde

        Args:
            dataset (list[str]): list of strings/text.
            verbose (bool): print progress/info.

        Returns:
            list[list[int]]: tokenized_dataset - [list of encoded tokens for sample in dataset].
        """
        self.int2tiktoken = set()
        tokenized_dataset = []

        for example in dataset:
            if example:
                tokens = self.tokenizer.encode(example)
                self.int2tiktoken.update(tokens)
                tokenized_dataset.append(tokens)

        self.int2tiktoken = np.array(list(self.int2tiktoken))
        self.vocab_size = len(self.int2tiktoken)
        if verbose: print(f'vocab size = {self.vocab_size}')

        self.tiktoken2int = np.zeros(max(self.int2tiktoken) + 1, int)
        self.tiktoken2int[self.int2tiktoken] = np.arange(self.vocab_size)

        return tokenized_dataset

    def write_cache(self, directory, dataset=None, verbose=False):
        """ Functiont to write token lookups and encoded dataset into pickle files.
            author: girish d. hegde

        Args:
            directory (Path/str): path to cache directory.
            dataset (list[str]/list[list[int]]): list[str] dataset or list[list[int]] encoded tokenized dataset.
            verbose (bool): print progress/info.

        Returns:
            list[int]: dataset - encoded text dataset.
        """
        tokenized = not isinstance(dataset[0], str)
        if dataset is not None:
            if verbose: print('tokenizing dataset ... ')
            temp = []
            for i, example in enumerate(dataset):
                if example:
                    example = self.tokenizer.encode(example) if not tokenized else example
                    temp.append(self.tiktoken2int[example])
            dataset = np.hstack(temp)
            if verbose: print(f'total tokens in dataset = {dataset.shape[0]/1e6}M')

        if verbose: print('writing cache ...')
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        if dataset is not None:
            with open(directory/'dataset.pkl', 'wb') as fp: pickle.dump(dataset, fp)
        with open(directory/'int2tiktoken.pkl', 'wb') as fp: pickle.dump(self.int2tiktoken, fp)
        with open(directory/'tiktoken2int.pkl', 'wb') as fp: pickle.dump(self.tiktoken2int, fp)
        if verbose: print('caching done.')

        return dataset

    def read_cache(self, directory, load_dataset=True):
        """ Functiont to read cached tokenizer files.
            author: girish d. hegde

        Args:
            directory (Path/str): path to tokenizer cache directory.

        Returns:
            np.ndarray[int]: int2tiktoken - encodings to tiktoken encodings table.
            np.ndarray[int]: tiktoken2int - tiktoken encodings to encodings table.
            list[str]/list[int]/None: dataset of list of strings or encoded dataset or None.
        """
        directory = Path(directory)
        dataset = None
        if load_dataset and (directory/'dataset.pkl').is_file():
            with open(directory/'dataset.pkl', 'rb') as fp: dataset = pickle.load(fp)
        with open(directory/'int2tiktoken.pkl', 'rb') as fp: self.int2tiktoken = pickle.load(fp)
        with open(directory/'tiktoken2int.pkl', 'rb') as fp: self.tiktoken2int = pickle.load(fp)
        return self.int2tiktoken, self.tiktoken2int, dataset

    def run(self, directory, dataset, verbose=False):
        """ Function run tokenization pipeline on text dataset file.
                1. read dataset into string text.
                2. convert dataset into tokens.
                3. get tokens to integer and vice versa lookups.
                4. cache token lookups and dataset.
            author: girish d. hegde

        Args:
            verbose (bool): print info.

        Returns:
            list[int]: dataset - encoded tokenized dataset.
        """
        dataset = self.read_dataset(dataset)
        dataset = self.tokenize(dataset, verbose)
        dataset = self.write_cache(directory, dataset, verbose)
        return dataset

    def encode(self, text):
        """ Function to encode text into tokens.
            author: girish d. hegde

        Args:
            text (str): input text string.

        Returns:
            np.ndarray[int]: tokens - list encoded text.
        """
        tokens = self.tokenizer.encode(text)
        tokens = self.tiktoken2int[tokens]
        return tokens

    def decode(self, tokens):
        """ Function to decode tokens into text/string.
            author: girish d. hegde

        Args:
            tokens (np.ndarray[int]/list[int]): list of enocded tokens.

        Returns:
            str: text - decoded text/string.
        """
        tokens = self.int2tiktoken[tokens]
        text = self.tokenizer.decode(tokens)
        return text


# TODO:
# add special task tokens
# prepare wikitext.py
# prepare codeparrot.py
# pretrain dataloader
# finetune dataloader
# collate functions