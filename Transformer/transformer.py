"""
Refs:
    https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    https://github.com/lucidrains/x-transformers
    http://peterbloem.nl/blog/transformers
    https://jalammar.github.io/illustrated-transformer/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


__author__ = "__Girish_Hegde__"


class Attention(nn.Module):
    """ Scaled Dot-Product Attention.
        author: girish d. hegde

    Q, K, V = (q x Wq), (k x Wk), (v x Wv)
    Attention(Q, K, V) = softmax((Q x K.T)/(sqrt(dk))) x V

    Args:
        d_model (int): dimension.
        h (int): number of heads. (dq = dk = dv = d = d_model/h).
        dropout (float): dropout probability.
    """
    def __init__(self, d_model, h=1, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d = d_model//h
        self.to_q = nn.Linear(d_model, self.d, bias=False)
        self.to_k = nn.Linear(d_model, self.d, bias=False)
        self.to_v = nn.Linear(d_model, self.d, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, q, k=None, v=None, mask=None):
        """ Forward.
            author: girish d. hegde

        Args:
            q (torch.tensor): [bs, seq_len, d_model] - input tensor.
            k (torch.tensor): [bs, seq_len, d_model] - input tensor.
            v (torch.tensor): [bs, seq_len, d_model] - input tensor.
            mask (torch.tensor[bool]): [seq_len, seq_len] - boolean mask.

        Returns:
            torch.tensor: [bs, seq_len, d] - output attention.
        """
        k = q if k is None else k
        v = q if v is None else v
        Q, K, V = self.to_q(q), self.to_k(k), self.to_v(v)  # [bs, seq_len, d]
        QK = torch.bmm(Q, K.permute(0, 2, 1))/(self.d ** 0.5)  # [bs, seq_len, seq_len]
        if mask is not None:
            QK[:, mask] = -float('inf')
        QK = F.softmax(QK, dim=-1)
        QK = self.dropout(QK) if self.dropout is not None else QK
        attention = torch.bmm(QK, V)  # [bs, seq_len, d]
        return attention
