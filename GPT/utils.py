from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


def save_checkpoint(
        net, epoch, loss, best, filename, **kwargs,
    ):
    ckpt = {
        'net':{
            'kwargs':net.get_init_params(),
            'state_dict':net.state_dict(),
        },
        'training':{
            'epoch':epoch, 'loss':loss, 'best':best,
        },
        'kwargs':kwargs,
    }
    torch.save(ckpt, filename)
    return ckpt


def load_checkpoint(filename, net, device='cpu'):
    epoch, best = 0, float('inf')
    kwargs = None
    if filename is not None:
        if Path(filename).is_file():
            ckpt = torch.load(filename, map_location=device)
            net.load_state_dict(ckpt['net']['state_dict'])
            print('Checkpoint loaded successfully ...')
            if 'training' in ckpt:
                epoch, loss, best = ckpt['training'].values()
                print('Training parameters loaded successfully ...')
            if 'kwargs' in ckpt:
                kwargs = ckpt['kwargs']
                print('Additional kwargs loaded successfully ...')
    return net, epoch, best, kwargs


# @torch.no_grad()
# def logits2text(logits, int2token, ):
#     """ Function to convert model prediction logits into words.
#         author: girish d. hegde

#     Args:
#         logits (torch.tensor[float]): [seq_len, vocab_size] - logits or [seq_len, ] - positions.
#         int2token (dict[int:str]): [vocab_size, ] - int to token lookup.

#     Returns:
#         str: output string(set of words).
#     """
#     if logits.ndimension() == 2:
#         positions = torch.argmax(logits.detach().cpu(), dim=-1)
#     else:
#         positions = logits.detach().cpu().type(torch.int32)
#     words = [int2token[p.item()] for p in positions]
#     return ' '.join(words)


# def write_pred(input_, logits, in_int2tk, out_int2tk, filename, label=''):
#     in_text = logits2text(input_, in_int2tk)
#     out_text = logits2text(logits, out_int2tk)
#     data = f'\n{"-"*100}\n{label}\n{"-"*100}\nIN: {in_text}\nOUT: {out_text}\n{"-"*100}'
#     with open(filename,'a' if Path(filename).is_file() else 'w', encoding="utf-8") as fp:
#          fp.write(data)
#     return in_text, out_text

# @torch.no_grad()
# def sample(
#         inp, net, tokenizer,
#         in_int2tk, out_int2tk,
#         start_token='<S>', end_token='<E>',
#         pad_token='<P>', ukn_token='<U>',
#         top_k=1, max_size=100, device='cpu'
#     ):
#     """ Function to sample output text from trained model.
#         author: girish d. hegde

#     Args:
#         inp (str): Any input sentence.
#         net (torch.nn.Module): trained model.
#         tokenizer (callable object/function): input sentence tokenizer.
#         in_int2tk (dict[int:str]): input integer to token lookup.
#         out_int2tk (dict[int:str]): target integer to token lookup.
#         start_token (str): start of sentence indication token.
#         end_token (str): end of sentence indication token.
#         pad_token (str): padding place holder indication token.
#         ukn_token (str): token for unknown words in the vocab.
#         top_k (int): topk sampling.
#         max_size (int): Max output tokens/words size.
#         device (torch.device): cpu or cuda.

#     Returns:
#         str: output string/text.
#     """
#     in_tk2int = {tk: i for i, tk in in_int2tk.items()}
#     out_tk2int = {tk: i for i, tk in out_int2tk.items()}
#     start_value, end_value = out_tk2int[start_token], out_tk2int[end_token]

#     net = net.to(device)
#     net.eval()

#     # Get tokens, add start & end token, fill unknown tokens if any.
#     inp = [start_token] + [str(tk) for tk in tokenizer(inp)] + [end_token]
#     inp = [tk if tk in in_tk2int else ukn_token for tk in inp]
#     inp = [in_tk2int[tk] for tk in inp]

#     output = net.generate(
#         inp, start_value, end_value, top_k, None, max_size
#     )
#     output = [out_int2tk[tk] for tk in output if tk != end_value]

#     return ' '.join(output)

