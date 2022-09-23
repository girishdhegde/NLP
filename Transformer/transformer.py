"""
Refs:
    https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    https://github.com/lucidrains/x-transformers
    http://peterbloem.nl/blog/transformers
    https://jalammar.github.io/illustrated-transformer/
"""


from email.errors import HeaderMissingRequiredValue
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


__author__ = "__Girish_Hegde__"


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()


class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()


class Attention(nn.Module):
    """ Scaled Dot-Product Attention.
        author: girish d. hegde

    Q, K, V = (q x Wq), (k x Wk), (v x Wv)
    Attention(Q, K, V) = softmax((Q x K.T)/(sqrt(dk))) x V

    Args:
        emb_dim (int): dimension.
        h (int): number of heads. (dq = dk = dv = d = emb_dim/h).
        dropout (float): dropout probability.
    """
    def __init__(self, emb_dim, h=1, dropout=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.h = h
        self.d = emb_dim//h
        self.to_q = nn.Linear(emb_dim, self.d, bias=False)
        self.to_k = nn.Linear(emb_dim, self.d, bias=False)
        self.to_v = nn.Linear(emb_dim, self.d, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, q, k=None, v=None, mask=None):
        """ Forward.
            author: girish d. hegde

        Args:
            q (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
            k (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
            v (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
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
        emb_dim (int): dimension.
        heads (int): number of heads. (dq = dk = dv = d = emb_dim/h).
        bias (bool): adde bias to input and output projection layers.
        act (nn.Module): output projection activation function.
        dropout (float): dropout probability.
    """
    def __init__(self, emb_dim, heads=1, bias=False, act=nn.Identity, dropout=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.d = emb_dim//heads

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.to_k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.to_v = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, q, k=None, v=None, mask=None):
        """ Forward.
            author: girish d. hegde

        Args:
            q (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
            k (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
            v (torch.tensor): [bs, seq_len, emb_dim] - input tensor.
            mask (torch.tensor[bool]): [seq_len, seq_len] - boolean mask.

        Returns:
            torch.tensor: [bs, seq_len, emb_dim] - output values.
            torch.tensor: [bs, heads, seq_len, seq_len] - attention.
        """
        k = q if k is None else k
        v = q if v is None else v

        Q, K, V = self.to_q(q), self.to_k(k), self.to_v(v)  # [bs, seq_len, emb_dim]
        Q, K, v = (rearrange(T, 'b l (h d) -> (b h) l d', h=self.heads) for T in (Q, K, V))

        attn = torch.bmm(Q, K.permute(0, 2, 1))/(self.d ** 0.5)  # [bs*heads, seq_len, seq_len]
        if mask is not None:
            attn[:, mask] = -float('inf')
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) if self.dropout is not None else attn

        out = torch.bmm(attn, V)  # [bs*heads, seq_len, d]
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.heads)  # [bs, seq_len, emb_dim]
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
        y, *_ = self.module(x)
        return self.layernorm(x + y)


class FFN(nn.Module):
    """ FeedForward Network
        author: girish d. hegde

    Args:
        emb_dim (int): dimension.
        act (nn.Module): activation function.
    """
    def __init__(self, emb_dim, act=nn.ReLU):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff = nn.Sequential([
            nn.Linear(emb_dim, 4*emb_dim),
            act(),
            nn.Linear(4*emb_dim, emb_dim)
        ])

    def forward(self, x):
        return self.ff(x)


class Encoder(nn.Module):
    def __init__(self, emb_dim, heads, num_layers=2, attn_act=nn.Identity, ffn_act=nn.ReLU, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.num_layers = num_layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                SubLayer(MHA(emb_dim, heads, bias=False, act=attn_act, dropout=dropout), emb_dim)
            )
            self.layers.append(SubLayer(FFN(emb_dim, act=ffn_act), emb_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()


