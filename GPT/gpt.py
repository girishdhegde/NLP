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
        self.post_act = post_act or nn.Identity()
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
        attn = self.attn_dropout(attn)

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
        x = x + self.attn(self.pre_attn_ln(x), mask=mask)[0]
        x = x + self.ffn(self.pre_ffn_ln(x))
        return x


class GPT(nn.Module):
    """ GPT2 model
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
        self.emb_drop = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *[Block(emb_dim, heads, attn_dropout, res_dropout) for _ in range(num_layers)]
        )
        self.transformer_ln = nn.LayerNorm(emb_dim)
        self.to_logits = nn.Linear(emb_dim, vocab_size, bias=False)

        # weight tying: https://github.com/karpathy/nanoGPT/blob/e0c689cf38478eea9416757cec5f834620983862/model.py#L122
        self.emb.weight = self.to_logits.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
        x = self.emb_drop(x)
        x = self.transformer(x)
        x = self.transformer_ln(x)
        x = self.to_logits(x)
        return x

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

    @property
    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        # logit_params = sum(p.numel() for p in self.to_logits.parameters())
        return total

    @classmethod
    def create_from_ckpt(cls, filename):
        ckpt = torch.load(filename)
        net = cls(**ckpt['config'])
        net.load_state_dict(ckpt['state_dict'])
        return net

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        return

    def get_optimizer(self, lr=2.5e-4, betas=(0.9, 0.999), weight_decay=0.01):
        """ Function to get AdamW optimizer with no weight decays for certain params.
            (no_decays -> bias, layernorm, embedding layers weights)
            author: girish d. hegde

        Refs:
            https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L215
            https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/6

        Args:
            lr (float): learning rate.
            betas (tuple[float]): betas.
            weight_decay (float): L2 regularization weight decay.

        Returns:
            torch.optim.AdamW : AdamW optimizer.
        """
        # separate weight decay and non weight decay parameters
        blacklist = ('bias', 'ln', 'emb')  # => blacklist = (torch.nn.LayerNorm, torch.nn.Embedding)
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            decayable = True
            for blk_str in blacklist:
                if blk_str in name:
                    no_decay.append(param)
                    decayable = False
                    break
            if decayable:
                decay.append(param)

        # validate that all parameters are considered
        all_params = set(self.parameters())
        set_decay, set_no_decay = set(decay), set(no_decay)
        inter_params = set_decay & set_no_decay
        union_params = set_decay | set_no_decay
        assert len(decay) == len(set_decay), "duplicates found in decay params"
        assert len(no_decay) == len(set_no_decay), "duplicates found in no decay params"
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(all_params - union_params) == 0, "parameters were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, indices, max_new_tokens, temperature=1.0, top_k=None, end_token=None):
        """ Function to generate tokens from model

        Args:
            indices (torch.LongTensor): [N, ] - input encoded token indices from tokenizer.
            max_new_tokens (int): generate max_new_tokens.
            temperature (float): controls randomness. 1 -> as it is(random), 0 -> precise.
            top_k (int): top_k sampling. provides diversity.
            end_token (int): end generation token.

        Refs:
            https://github.com/karpathy/nanoGPT/blob/master/model.py

        Returns:
            torch.LongTensor: indices - [N + max_new_tokens] - generated tokens.
            torch.LongTensor: pred - [max_new_tokens] - generated new tokens.
        """
        self.eval()
        indices = indices[None, :] if len(indices.shape) == 1 else indices
        prompt_len = indices.shape[1]
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx = indices if indices.size(1) <= self.context_size else indices[:, -self.context_size:]
            logits = self(idx)
            logits = logits[:, -1, :]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            tk = torch.multinomial(probs, num_samples=1)
            if tk.item() == end_token: break
            indices = torch.cat((indices, tk), dim=1)
        self.train()
        indices = indices.squeeze()
        return indices, indices[prompt_len:]