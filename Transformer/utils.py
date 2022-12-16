from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def save_checkpoint(
        in_int2tk, out_int2tk,
        start_token, end_token, pad_token, ukn_token,
        net, epoch, loss, best, filename,
    ):
    ckpt = {
        'dataset':{
            'in_int2tk':in_int2tk, 'out_int2tk':out_int2tk,
            'start_token':start_token, 'end_token':end_token,
            'pad_token':pad_token, 'ukn_token':ukn_token,
        },
        'net':{
            'kwargs':net.get_init_params(),
            'state_dict':net.state_dict(),
        },
        'training':{
            'epoch':epoch, 'loss':loss, 'best':best,
        }
    }
    torch.save(ckpt, filename)
    return ckpt


def load_checkpoint(filename, net, device='cpu'):
    epoch, best = 0, float('inf')
    in_int2tk, out_int2tk = None, None
    start_token, end_token = None, None
    pad_token, ukn_token = None, None
    if filename is not None:
        if Path(filename).is_file():
            ckpt = torch.load(filename, map_location=device)
            net.load_state_dict(ckpt['net']['state_dict'])
            print('Checkpoint loaded successfully ...')
            if 'training' in ckpt:
                epoch, loss, best = ckpt['training'].values()
                print('Training parameters loaded successfully ...')
            if 'dataset' in ckpt:
                in_int2tk, out_int2tk, start_token, end_token, pad_token, ukn_token = ckpt['dataset'].values()
                print('Dataset parameters loaded successfully ...')
    return net, epoch, best, in_int2tk, out_int2tk, start_token, end_token, pad_token, ukn_token


@torch.no_grad()
def logits2text(logits, int2token, ):
    """ Function to convert model prediction logits into words.
        author: girish d. hegde

    Args:
        logits (torch.tensor[float]): [seq_len, vocab_size] - logits or [seq_len, ] - positions.
        int2token (dict[int:str]): [vocab_size, ] - int to token lookup.

    Returns:
        str: output string(set of words).
    """
    if logits.ndimension() == 2:
        positions = torch.argmax(logits.detach().cpu(), dim=-1)
    else:
        positions = logits.detach().cpu().type(torch.int32)
    words = [int2token[p.item()] for p in positions]
    return ' '.join(words)


def write_pred(input_, logits, in_int2tk, out_int2tk, filename, label=''):
    in_text = logits2text(input_, in_int2token)
    out_text = logits2text(logits, out_int2token)
    data = f'\n{"-"*100}\n{label}\n{"-"*100}\nIN: {in_text}\nOUT: {out_text}\n{"-"*100}'
    with open(filename,'a' if Path(filename).is_file() else 'w') as fp:
         fp.write(data)
    return text