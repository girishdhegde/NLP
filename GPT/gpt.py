"""
Refs:
    https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
    https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    https://github.com/karpathy/minGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


__author__ = "__Girish_Hegde__"


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
        pre_act (Callable): pre attention projection activation. (applied on Q, K, V).
        post_act (Callable): output projection activation. (applied after attn).
        attn_dropout (float): attention dropout probability.
        res_dropout (float): residual dropout probability (applied after last projection).
    """
    def __init__(self, emb_dim, heads=1, bias=False, pre_act=None, post_act=None, attn_dropout=None, res_dropout=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.d = emb_dim//heads

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.to_k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.to_v = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.pre_act = pre_act or nn.Identity()
        self.post_act = post_ac or nn.Identity()
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout is not None else nn.Identity()
        self.res_dropout = nn.Dropout(res_dropout) if res_dropout is not None else nn.Identity()

    def forward(self, q, k=None, v=None, mask=None):
        """ Forward.
            author: girish d. hegde

        Args:
            q (torch.tensor): [bs, seq_len, emb_dim] - query input tensor.
            k (torch.tensor): [bs, seq_len, emb_dim] - key input tensor.
            v (torch.tensor): [bs, seq_len, emb_dim] - value input tensor.
            mask (torch.tensor[bool]): [seq_len, seq_len] - boolean mask. attn[where mask is True] = -inf.

        Returns:
            torch.tensor: [bs, seq_len, emb_dim] - output values.
            torch.tensor: [bs, heads, seq_len, seq_len] - attention.
        """
        k = q if k is None else k
        v = q if v is None else v

        Q, K, V = self.to_q(q), self.to_k(k), self.to_v(v)  # [bs, seq_len, emb_dim]
        Q, K, V = (self.pre_act(T) for T in (Q, K, V))
        Q, K, V = (rearrange(T, 'b l (h d) -> (b h) l d', h=self.heads) for T in (Q, K, V))

        attn = torch.bmm(Q, K.permute(0, 2, 1))/(self.d ** 0.5)  # [bs*heads, seq_len, seq_len]
        if mask is not None:
            attn[..., mask] = -float('inf')
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, V)  # [bs*heads, seq_len, d]
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.heads)  # [bs, seq_len, emb_dim]
        out = self.proj(out)
        out = self.post_act(out)
        out = self.res_dropout(out)

        return out, rearrange(attn, '(b h) i j -> b h i j', h=self.heads)


class Block(nn.Module):
    """ GPT2 Block
        author: girish d. hegde

    Args:
        emb_dim (int): dimension.
        heads (int): number of heads. (dq = dk = dv = d = emb_dim/h).
        attn_dropout (float): attention dropout probability.
        res_dropout (float): residual dropout probability (applied after last projection).
    """
    def __init__(self, emb_dim, heads, attn_dropout=None, res_dropout=None):
        super().__init__()
        self.emb_dim = emb_dim

        self.pre_attn_ln = nn.LayerNorm(emb_dim)
        self.attn = MHA(emb_dim, heads, attn_dropout=attn_dropout, res_dropout=res_dropout)
        self.pre_ffn_ln = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.GELU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(res_dropout) if res_dropout is not None else nn.Identity(),
        )

    def forward(self, x):
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
        x = x + self.attn(self.pre_attn_ln(x), mask)
        x = x + self.ffn(self.pre_ffn_ln(x))
        return x


class GPT(nn.Module):
    """ GPT
        author: girish d. hegde

    Args:
        emb_dim (int): d_model = embedding dimension.
        heads (int): number of heads per MHA layer.
        num_layers (int): number of layers.
        vocab_size (int): vocabulary size.
        context_size (int): context size.
        emb_dropout (float): embedding dropout probability.
        attn_dropout (float): attention dropout probability.
        res_dropout (float): residual dropout probability (applied after last projection).
    """
    def __init__(self,
            emb_dim, heads, num_layers,
            vocab_size, context_size,
            emb_dropout=0.1, attn_dropout=0.1, res_dropout=0.1,
        ):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.heads = heads
        self.num_layers = num_layers
        self.emb_dropout = emb_dropout
        self.attn_dropout = attn_dropout
        self.res_dropout = res_dropout

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_size, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *[Block(emb_dim, heads, attn_dropout, res_dropout) for _ in range(num_layers)]
        )
        self.transformer_ln = nn.LayerNorm(emb_dim)
        self.to_logits = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """ GPT forward
            author: girish d. hegde

        Args:
            x (torch.tensor[torch.int64]): [bs, seq_len] - token encodings.

        Returns:
            torch.tensor[torch.float32]: [bs, seq_len, vocab_size] - target logits.
        """
        pos = torch.arange(0, x.shape[1], dtype=torch.int64, device=x.device)[None, :]
        x = self.emb(x) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = self.transformer_ln(x)
        x = self.to_logits(x)
        return x

    @torch.no_grad()
    def generate(self, ):
        return

    def get_config(self):
        config = {
            'emb_dim':self.emb_dim, 'heads':self.heads, 'num_layers':self.num_layers,
            'vocab_size':self.vocab_size, 'context_size':self.context_size,
            'emb_dropout':self.emb_dropout, 'attn_dropout':self.attn_dropout, 'res_dropout':self.res_dropout,
        }
        return config

    def save_ckpt(self, filename):
        ckpt = {
            'config':self.get_config(),
            'state_dict':self.state_dict(),
        }
        torch.save(ckpt, filename)
        return ckpt

    @classmethod
    def create_from_ckpt(cls, filename):
        ckpt = torch.load(filename)
        net = cls(**ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        return net
