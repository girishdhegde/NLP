import re
import random
from pathlib import Path
from copy import deepcopy
import pickle

import numpy as np
import tiktoken  # openai gpt2 bpe tokenizer
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class BPETokenizer(tiktoken.core.Encoding):
    """ open-ai gpt2 bpe tokenizer.
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
        n_tasks (int): add task special tokens - ['<|task|>' for task in range(n_tasks)].

    Refs:
        https://github.com/openai/tiktoken
        https://github.com/openai/tiktoken/issues/9
    """
    def __init__(self, n_tasks=0):
        sp_tkns = {"<|endoftext|>": 50256}
        for i in range(n_tasks):
            sp_tkns[f"<|{i + 1}|>"] = 50257 + i
        mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
            vocab_bpe_file="az://openaipublic/gpt-2/encodings/main/vocab.bpe",
            encoder_json_file="az://openaipublic/gpt-2/encodings/main/encoder.json",
        )
        kwargs = {
            "name": "gpt2",
            "explicit_n_vocab": 50257 + n_tasks,
            "pat_str": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            "mergeable_ranks": mergeable_ranks,
            "special_tokens": sp_tkns,
        }
        super().__init__(**kwargs)
        self.n_tasks = n_tasks

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
                tokens = self.encode(example)
                tokens = torch.tensor(tokens, dtype=torch.int64) if to_torch else tokens
                tokenized_dataset.append(tokens)
                ntokens += len(tokens)

        if verbose: print(f'vocab size = {self.n_vocab}')
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
        tokens = super().encode(text, allowed_special='all')
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
        text = super().decode(tokens)
        return text


class PretrainSet(Dataset):
    """ Pytorch Dataset class for text corpus pretraining of GPT.
        author: girish d. hegde

    Args:
        dataset (torch.LongTensor/list[list[int]]): Encoded tokenized dataset.
        block_size (int): sequence length.
    """
    def __init__(self, dataset, block_size=512):
        super().__init__()
        self.block_size = block_size
        if not isinstance(dataset, torch.Tensor):
            self.dataset = torch.hstack([torch.tensor(data, dtype=torch.int64) for data in dataset])
        else:
            self.dataset = dataset
        self.max_start = len(self.dataset) - (block_size + 1)
        self.len = self.max_start

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Returns:
            torch.LongTensor: [block_size, ] - input tokens encoding.
            torch.LongTensor: [block_size, ] - ouput tokens encoding.
        """
        start = random.randint(0, self.max_start)
        end = start + self.block_size
        return self.dataset[start:end], self.dataset[start + 1:end + 1]


# TODO:
# finetune dataloader and collate functions if required