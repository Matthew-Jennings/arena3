""" "_summary_"
Functionality here is copied from my original implementations in notebooks while
completing the course.
"""

from dataclasses import dataclass

import einops
from jaxtyping import Float, Int
import torch as t
from torch import Tensor
import torch.nn as nn
from transformer_lens.utils import gelu_new


device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)


@dataclass
class Config:
    d_head: int = 64
    d_mlp: int = 3072
    d_model: int = 768
    d_vocab: int = 50257
    debug: bool = True
    init_range: float = 0.02
    layer_norm_eps: float = 1e-5
    n_ctx: int = 1024
    n_heads: int = 12
    n_layers: int = 12


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(dim=-1, keepdim=True)
        var = residual.var(dim=-1, keepdim=True, unbiased=False)

        res = (residual - mean) / (var + self.cfg.layer_norm_eps).sqrt()

        return res * self.w + self.b


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        # W_pos has a maximum length of n_ctx. If seq_len is less than n_ctx, we only take the first seq_len positions
        # W_pos is the same for all batches, so we repeat it along the batch dimension
        return self.W_pos[:seq_len].repeat(batch, 1, 1)
        # return einops.repeat(self.W_pos[:seq_len], "position d_model -> batch position d_model", batch=batch)


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer(
            "IGNORE", t.tensor(float("-inf"), device=device, dtype=t.float32)
        )

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        mask = t.triu(
            t.ones(attn_scores.shape[-2:], device=attn_scores.device), diagonal=1
        )
        # remember, broadcasting starts with trailing dimensions
        return attn_scores.masked_fill(mask == 1, self.IGNORE)

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        batch, seq_len, d_model = normalized_resid_pre.shape

        Q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_Q
        )
        K = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_K
        )
        V = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_V
        )

        attn_scores = einops.einsum(
            Q,
            K,
            "batch posn_Q n_heads d_head, batch posn_K n_heads d_head -> batch n_heads posn_Q posn_K",
        )
        attn_scores_scaled = attn_scores / (self.cfg.d_head**0.5)
        attn_scores_scaled_masked = self.apply_causal_mask(attn_scores_scaled)

        attn_probs = attn_scores_scaled_masked.softmax(dim=-1)

        z = einops.einsum(
            V,
            attn_probs,
            "batch posn_K n_heads d_head, batch n_heads posn_Q posn_K -> batch posn_Q n_heads d_head",
        )

        return (
            einops.einsum(
                z,
                self.W_O,
                "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model",
            )
            + self.b_O
        )


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        h = gelu_new(
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
            )
            + self.b_in
        )
        return (
            einops.einsum(
                h, self.W_out, "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
            )
            + self.b_out
        )


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        normalised_resid_pre = self.ln1(resid_pre)
        attn_output = self.attn(normalised_resid_pre)
        resid_mid = resid_pre + attn_output

        normalised_resid_mid = self.ln2(resid_mid)
        mlp_output = self.mlp(normalised_resid_mid)
        resid_post = resid_mid + mlp_output

        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)

        resid = embed + pos_embed

        for block in self.blocks:
            resid = block(resid)

        resid_final = self.ln_final(resid)
        return self.unembed(resid_final)


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


@dataclass
class TransformerTrainingArgs:
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
