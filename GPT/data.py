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
    """ input output sentence word level tokenization
        author: girish d. hegde

    text data corpus format:
        filename.txt
            <in_token> input sentence 1. <out_token> output sentence 1.
            <in_token> input sentence 2. <out_token> output sentence 2.
            ...
            ...
            <in_token> input sentence n. <out_token> output sentence n.

    Refs:
        https://github.com/brendenlake/SCAN
        https://towardsdatascience.com/dynamic-word-tokenization-with-regex-tokenizer-801ae839d1cd
    """
    def __init__(self, ):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def read(self, filename):
        """ Fuction to read pickle file containing text.
            author: girish d. hegde

        Args:
            filename (Path): path to .txt file.

        Returns:
            str: corpus of text
        """
        with open(filename, 'rb') as fp:
            corpus = pickle.load(fp)
        return corpus

    def tokenize(self, dataset, verbose=False):
        """ Fucntion to tokenize corpus of text to input output sentence tokens/words.
            author: girish d. hegde

        Args:
            corpus (str/generator): text corpus.
            in_token (str): token representing input sentence.
            out_token (str): token representing output sentence.
            start_token (str): start of sentence indication token.
            end_token (str): end of sentence indication token.
            lowercase (bool): convert all tokens to lowercase.

        Returns:
            str: corpus with special characters space padded.
            list[list[str]]: [list of input tokens/words of sentence for sentence in corpus].
            list[list[str]]: [list of output tokens/words of sentence for sentence in corpus].
            list[str]: unique input tokens.
            list[str]: unique output tokens.
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

    def write_cache(self, directory, dataset=None, tokenized=True, verbose=False):
        """ Functiont to write integer, tiktokens lookups and encoded dataset into pickle files.
            author: girish d. hegde

        Args:
            directory (Path/str): path to output directory.
        """
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
            dict[int:str]: in_int2tk - integer to token lookup of input sentences.
            dict[int:str]: in_int2tk - integer to token lookup of output sentences.
        """
        directory = Path(directory)
        dataset = None
        if load_dataset and (directory/'dataset.pkl').is_file():
            with open(directory/'dataset.pkl', 'rb') as fp: dataset = pickle.load(fp)
        with open(directory/'int2tiktoken.pkl', 'rb') as fp: int2tiktoken = pickle.load(fp)
        with open(directory/'tiktoken2int.pkl', 'rb') as fp: tiktoken2int = pickle.load(fp)
        return int2tiktoken, tiktoken2int, dataset

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
            np.ndarray[int]: dataset - encoded tokenized dataset.
        """
        dataset = self.read(dataset, encoding)
        dataset = self.tokenize(dataset, verbose)
        dataset = self.write_cache(directory, dataset, True, verbose)
        return dataset

    def encode(self, text):
        """ Function to encode text into tokens.
            author: girish d. hegde

        Args:
            text (str): input text string.

        Returns:
            np.ndarray[int]: tokens - list encoded text.
        """
        tokens = self.tokenizer(text)
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