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
    def run(
            cls, filename, in_token='IN:', out_token='OUT:', start_token='<S>', end_token='<E>',
            pad_token='<P>', ukn_token='<U>',
            lowercase=True, out_json=None, encoding='utf-8', verbose=False,
        ):
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
            start_token (str): start of sentence indication token.
            end_token (str): end of sentence indication token.
            pad_token (str): padding place holder indication token.
            ukn_token (str): token for unknown words in the vocab.
            lowercase (bool): convert all tokens to lowercase.
            out_json (Path): path to .json file for writing int2token lookup.
            encoding (str): input file encoding.
            verbose (bool): print info.

        Returns:
            list[list[str]]: in_corpus - [list of input tokens/words of sentence for sentence in corpus].
            list[list[str]]: out_corpus - [list of output tokens/words of sentence for sentence in corpus].
            dict[int:str]: in_int2tk - integer to token lookup of input sentences.
            dict[int:str]: out_int2tk - integer to token lookup of output sentences.
        """
        corpus = cls.read(filename, encoding)
        if verbose: print('Total characters in corpus = ', len(corpus))
        _, in_corpus, out_corpus, in_tokens, out_tokens = cls.tokenize(
            corpus, in_token, out_token, start_token, end_token, lowercase
        )
        in_tokens = in_tokens + [pad_token, ukn_token]
        out_tokens = out_tokens + [pad_token, ukn_token]
        if verbose: print('Total input sentence unique tokens = ', len(in_tokens))
        if verbose: print('Total output sentence unique tokens = ', len(out_tokens))
        if verbose: print('Total sentences = ', len(in_corpus))
        in_int2tk, in_tk2int = cls.get_lookups(in_tokens)
        out_int2tk, out_tk2int = cls.get_lookups(out_tokens)
        if out_json is not None: cls.write_json(out_json, in_int2tk, out_int2tk)
        return in_corpus, out_corpus, in_int2tk, out_int2tk


class TranslationSet(Dataset):
    """ Pytorch word level Dataset class for input output sentence pairs text.
        author: girish d. hegde

    Args:
        in_corpus (list[list[str]]): [list of input tokens/words of sentence for sentence in corpus].
        out_corpus (list[list[str]]): [list of output tokens/words of sentence for sentence in corpus].
        in_int2tk (dict[int:str]): integer to token lookup of input sentences.
        out_int2tk (dict[int:str]): integer to token lookup of output sentences.
        device (torch.device/str):  torch data device 'cuda' or 'cpu'.
    """
    def __init__(self, in_corpus, out_corpus, in_int2tk, out_int2tk, device='cuda'):
        super().__init__()
        in_tk2int = {tk:i for i, tk in in_int2tk.items()}
        out_tk2int = {tk:i for i, tk in out_int2tk.items()}
        self.in_vocab_size = len(in_int2tk)
        self.out_vocab_size = len(out_int2tk)
        self.in_enc = [
            torch.tensor([in_tk2int[tk] for tk in sentence], dtype=torch.int64, device=device)
            for sentence in in_corpus
        ]
        self.out_enc = [
            torch.tensor([out_tk2int[tk] for tk in sentence], dtype=torch.int64, device=device)
            for sentence in out_corpus
        ]
        self.len = len(in_corpus)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Returns:
            torch.tensor: [seq_len, ] - input tokens encoding.
            torch.tensor: [seq_len, ] - ouput tokens encoding.
        """
        return self.in_enc[index], self.out_enc[index]


class SeqCollater:
    """ SeqCollater - variable sequence length collate function creater.

    Args:
        inp_pad_value (int): input padding value.
        tgt_pad_value (int): target padding value.
    """
    def __init__(self, inp_pad_value=0, tgt_pad_value=0):
        self.inp_pad_value = inp_pad_value
        self.tgt_pad_value = tgt_pad_value

    def __call__(self, batch):
        """
        Args:
            batch (list[tuple[torch.tensor]]): [bs, ] - samples from __getitem__ dataset func.

        Returns:
            tuple[torch.tensor]:
                torch.tensor[int64]: [bs, max_seq_len] - same sequence size batched input data.
                torch.tensor[int64]: [bs, max_seq_len] - same sequence size batched target data.
        """
        inp = torch.nn.utils.rnn.pad_sequence(
            [data[0] for data in batch],
            batch_first=True, padding_value=self.inp_pad_value
        )

        tgt = torch.nn.utils.rnn.pad_sequence(
            [data[1] for data in batch],
            batch_first=True, padding_value=self.tgt_pad_value
        )
        return inp, tgt
