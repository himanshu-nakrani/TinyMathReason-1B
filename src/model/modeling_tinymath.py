import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

class TinyMathConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        intermediate_dim: int = 5632,
        num_layers: int = 22,
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,
        max_seq_len: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_attention_heads
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, config: TinyMathConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        xq = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV heads for GQA
        xk = torch.repeat_interleave(xk, self.num_kv_groups, dim=2)
        xv = torch.repeat_interleave(xv, self.num_kv_groups, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)

class SwiGLU(nn.Module):
    def __init__(self, config: TinyMathConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: TinyMathConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TinyMathReason(nn.Module):
    def __init__(self, config: TinyMathConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        self.freqs_cis = precompute_freqs_cis(
            config.head_dim, config.max_seq_len * 2, config.rope_theta
        )

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:T].to(h.device)

        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
