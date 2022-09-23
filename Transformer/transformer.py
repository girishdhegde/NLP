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
            torch.tensor: [bs, seq_len, d] - output values.
            torch.tensor: [bs, seq_len, seq_len] - attention.
        """
        k = q if k is None else k
        v = q if v is None else v
        Q, K, V = self.to_q(q), self.to_k(k), self.to_v(v)  # [bs, seq_len, d]
        attn = torch.bmm(Q, K.permute(0, 2, 1))/(self.d ** 0.5)  # [bs, seq_len, seq_len]
        if mask is not None:
            attn[:, mask] = -float('inf')
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) if self.dropout is not None else attn
        out = torch.bmm(attn, V)  # [bs, seq_len, d]
        return out, attn


class MHA(nn.Module):
    """ Multi Headed Scaled Dot-Product Attention.
        author: girish d. hegde

    MHA(Q, K, V) = Concat(head0, ..., headn) x W
    headi:
        Q, K, V = (q x Wq), (k x Wk), (v x Wv)
        Attention(Q, K, V) = softmax((Q x K.T)/(sqrt(dk))) x V

    Args:
        d_model (int): dimension.
        heads (int): number of heads. (dq = dk = dv = d = d_model/h).
        act (nn.Module): activation function.
        dropout (float): dropout probability.
    """
    def __init__(self, d_model, heads=1, act=nn.Identity, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d = d_model//heads

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.act = act()
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
            torch.tensor: [bs, seq_len, d_model] - output values.
            torch.tensor: [bs, heads, seq_len, seq_len] - attention.
        """
        k = q if k is None else k
        v = q if v is None else v

        Q, K, V = self.to_q(q), self.to_k(k), self.to_v(v)  # [bs, seq_len, d_model]
        Q, K, v = (rearrange(T, 'b l (h d) -> (b h) l d', h=self.heads) for T in (Q, K, V))

        attn = torch.bmm(Q, K.permute(0, 2, 1))/(self.d ** 0.5)  # [bs*heads, seq_len, seq_len]
        if mask is not None:
            attn[:, mask] = -float('inf')
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) if self.dropout is not None else attn

        out = torch.bmm(attn, V)  # [bs*heads, seq_len, d]
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.heads)  # [bs, seq_len, d_model]
        out = self.proj(out)

        return self.act(out), rearrange(attn, '(b h) i j -> b h i j', h=self.heads)


class SubLayer(nn.Module):
    """ SubLayer
        out = LayerNorm(x + module(x))
        author: girish d. hegde

    Args:
        module (nn.Module): MHA or FeedForward initialized object.
        dims (int/tuple[int]): Layer Normalization feature dimension.
    """
    def __init__(self, module, dims):
        self.module = module
        self.layernorm = nn.LayerNorm(dims)

    def forward(self, x):
        return self.layernorm(x + self.module(x))


class FeedForward(nn.Module):
    """ FeedForward
        author: girish d. hegde

    Args:
        d_model (int): dimension.
        act (nn.Module): activation function.
    """
    def __init__(self, d_model, act=nn.ReLU):
        super().__init__()
        self.d_model = d_model
        self.ff = nn.Sequential([
            nn.Linear(d_model, 4*d_model),
            act(),
            nn.Linear(4*d_model, d_model)
        ])

    def forward(self, x):
        return self.ff(x)