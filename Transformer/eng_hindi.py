import json

import numpy as np
from datasets import load_dataset
from spacy.lang.hi import Hindi
from spacy.lang.en import English


__author__ = "__Girish_Hegde__"


def tatoeba_en_hi(
        split='train',
        start_token='<S>', end_token='<E>',
        pad_token='<P>', ukn_token='<U>',
        out_json=None, verbose=False,
    ):
    """ Function for tatoeba english-hindi traslation tokenization.
        Tatoeba is a dataset of collection of sentences and translations.
        author: girish d. hegde

    Refs:
        https://huggingface.co/datasets/tatoeba

    Args:
        split (str): 'train' or 'test'.
        start_token (str): start of sentence indication token.
        end_token (str): end of sentence indication token.
        pad_token (str): padding place holder indication token.
        ukn_token (str): token for unknown words in the vocab.
        out_json (Path): path to .json file for writing int2token lookup.
        verbose (bool): print info.

    Returns:
        list[list[str]]: in_sentences - [list of input tokens/words of sentence for sentence english corpus].
        list[list[str]]: out_sentences - [list of output tokens/words of sentence for sentence hindi corpus].
        dict[int:str]: in_int2tk - integer to token lookup of input sentences.
        dict[int:str]: out_int2tk - integer to token lookup of output sentences.
    """
    dataset = load_dataset("tatoeba", lang1="en", lang2="hi")
    dataset = dataset[split]
    if verbose: print('tatoeba english-hindi dataset loaded ...')

    en_tokenizer = English()
    hi_tokenizer = Hindi()

    en_sentences = [en_tokenizer(data['translation']['en']) for data in dataset][:-1]
    en_sentences = [[start_token] + [str(tk) for tk in sentence] + [end_token] for sentence in en_sentences]
    en_tokens = [tk for sentence in en_sentences for tk in sentence] + [pad_token, ukn_token]
    en_tokens = np.unique(en_tokens)
    en_int2tk = dict(enumerate(en_tokens))
    if verbose: print(f'Total sentences = {len(en_sentences)}')
    if verbose: print(f'Total english tokens = {len(en_tokens)}')

    hi_sentences = [hi_tokenizer(data['translation']['hi']) for data in dataset][:-1]
    hi_sentences = [[start_token] + [str(tk) for tk in sentence] + [end_token] for sentence in hi_sentences]
    hi_tokens = [tk for sentence in hi_sentences for tk in sentence] + [pad_token + ukn_token]
    hi_tokens = np.unique(hi_tokens)
    hi_int2tk = dict(enumerate(hi_tokens))
    if verbose: print(f'Total hindi tokens = {len(hi_tokens)}')

    if out_json is not None:
        with open(out_json, 'w') as fp:
            json.dump([en_int2tk, hi_int2tk], fp, indent=4)
        if verbose: print(f'Token lookups written into json file ...')

    return en_sentences, hi_sentences, en_int2tk, hi_int2tk