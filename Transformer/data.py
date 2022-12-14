import re
import random
from pathlib import Path
from copy import deepcopy
import json

import torch
from torch.utils.data import Dataset


__author__ = "__Girish_Hegde__"


class InOutTokenizer:
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
    def __init__(self, ): pass

    @classmethod
    def read(cls, filename, encoding='utf-8'):
        """ Fuction to read file containing input output text.
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
    def tokenize(cls, corpus, in_token='IN:', out_token='OUT:', start_token='<S>', end_token='<E>', lowercase=True):
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
        sp_chars = re.sub('[A-Za-z]', '', deepcopy(corpus)).replace(' ', '')
        sp_chars = set(sp_chars)
        for sp in sp_chars:
            corpus = corpus.replace(sp, f' {sp} ')

        in_token = ''.join([f' {ch} ' if ch in sp_chars else ch for ch in in_token])
        out_token = ''.join([f' {ch} ' if ch in sp_chars else ch for ch in out_token])

        inout = (line.replace(in_token, '').split(out_token) for line in corpus.split('\n') if line.strip())

        def process(data):
            if lowercase:
                data = [d for d in data.strip().lower().split() if d != '']
            else:
                data = [d for d in data.strip().split() if d != '']
            return data

        in_, out = [], []
        for data in inout:
            in_.append([start_token] + process(data[0]) + [end_token])
            out.append([start_token] + process(data[1]) + [end_token])

        in_tokens = list(set(word for sentence in in_ for word in sentence))
        out_tokens = list(set(word for sentence in out for word in sentence))
        return corpus, in_, out, in_tokens, out_tokens

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
    def write_json(cls, filename, in_int2tk, out_int2tk):
        """ Functiont to write integer to tokens dictionary into json file.
            author: girish d. hegde

        Args:
            filename (Path): path to .json file.
            in_int2tk (dict[int:str]): integer to token lookup of input sentences.
            out_int2tk (dict[int:str]): integer to token lookup of output sentences.
        """
        with open(filename, 'w') as fp:
            json.dump([in_int2tk, out_int2tk], fp, indent=4)
        return True

    @classmethod
    def read_json(cls, filename):
        """ Functiont to read integer to tokens json file.
            author: girish d. hegde

        Args:
            filename (Path): path to .json file.

        Returns:
            dict[int:str]: in_int2tk - integer to token lookup of input sentences.
            dict[int:str]: in_int2tk - integer to token lookup of output sentences.
        """

        with open(filename) as fp:
            in_int2tk, out_int2tk = json.load(fp)
        in_int2tk = {int(k):v for k, v in in_int2tk.items()}
        out_int2tk = {int(k):v for k, v in out_int2tk.items()}
        return in_int2tk, out_int2tk

    @classmethod
    def run(cls, filename, in_token='IN:', out_token='OUT:', lowercase=True, out_json=None, encoding='utf-8', verbose=False):
        """ Function run tokenization pipeline on text corpus file.
                1. read file into string corpus.
                2. convert corpus into input output tokens.
                3. get tokens to integer and vice versa lookups.
                4. write token lookups.
            author: girish d. hegde

        Args:
            filename (Path): path to .txt file containg corpus of text data.
            in_token (str): token representing input sentence.
            out_token (str): token representing output sentence.
            lowercase (bool): convert all tokens to lowercase.
            out_json (Path): path to .json file for writing int2token lookup.
            encoding (str): input file encoding.
            verbose (bool): print info.

        Returns:
            list[str]: words - list of tokens/words in corpus.
            list[str]: tokens - unique tokens.
            dict[int:str]: int2token - integer to token lookup.
            dict[int:str]: token2int - token to integer lookup.
        """
        corpus = cls.read(filename, encoding)
        if verbose: print('Total characters in corpus = ', len(corpus))
        _, in_corpus, out_corpus, in_tokens, out_tokens = cls.tokenize(corpus, lowercase)
        if verbose: print('Total input sentence unique tokens = ', len(in_tokens))
        if verbose: print('Total output sentence unique tokens = ', len(out_tokens))
        if verbose: print('Total sentences = ', len(in_corpus))
        in_int2tk, in_tk2int = cls.get_lookups(in_tokens)
        out_int2tk, out_tk2int = cls.get_lookups(out_tokens)
        if out_json is not None: cls.write_json(out_json, in_int2tk, out_int2tk)
        return in_corpus, out_corpus, in_int2tk, out_int2tk


# class TextSet(Dataset):
#     """ Pytorch word level Dataset class for text corpus.
#         author: girish d. hegde

#     Args:
#         words (list[str]): list of tokens/words in corpus.
#         int2token (dict[int:str]): integer to token lookup.
#         seq_len (int): sequence length.
#         device (torch.device/str):  torch data device 'cuda' or 'cpu'.
#     """
#     def __init__(self, words, int2token, seq_len=50, device='cuda'):
#         super().__init__()
#         token2int = {tk:i for i, tk in int2token.items()}
#         self.vocab_size = len(int2token)
#         self.seq_len = seq_len
#         self.encoded = torch.tensor([token2int[token] for token in words], dtype=torch.int64, device=device)
#         # self.one_hot = torch.eye(self.vocab_size, dtype=torch.float32, device=device)
#         self.len = len(words) - (seq_len + 1)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, index):
#         """
#         Returns:
#             torch.tensor: [seq_len, ] - input tokens encoding.
#             torch.tensor: [seq_len, ] - ouput tokens encoding.
#         """
#         start = random.randint(0, self.len)
#         end = start + self.seq_len
#         # return self.encoded[start:end], self.one_hot[self.encoded[start + 1:end + 1]]
#         return self.encoded[start:end], self.encoded[start + 1:end + 1]
