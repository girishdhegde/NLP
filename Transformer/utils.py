from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def save_checkpoint(
        in_int2tk, out_int2tk, start_token, end_token,
        net, epoch, loss, best, filename,
    ):
    ckpt = {
        'dataset':{
            'in_int2tk':in_int2tk, 'out_int2tk':out_int2tk,
            'start_token':start_token, 'end_token':end_token,
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
    if filename is not None:
        if Path(filename).is_file():
            ckpt = torch.load(filename, map_location=device)
            net.load_state_dict(ckpt['net']['state_dict'])
            print('Checkpoint loaded successfully ...')
            if 'training' in ckpt:
                epoch, loss, best = ckpt['training'].values()
                print('Training parameters loaded successfully ...')
            if 'dataset' in ckpt:
                in_int2tk, out_int2tk, start_token, end_token = ckpt['dataset'].values()
                print('Dataset parameters loaded successfully ...')
    return net, epoch, best, in_int2tk, out_int2tk, start_token, end_token

