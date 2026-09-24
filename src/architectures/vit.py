import torch
from einops import pack, rearrange, repeat, unpack
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads=8, attention_head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = attention_head_dim * num_attention_heads
        project_out = not (num_attention_heads == 1 and attention_head_dim == embedding_dim)

        self.num_attention_heads = num_attention_heads
        self.scale = attention_head_dim**-0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, key_padding_mask=None):
        """key_padding_mask: (B, N), True marks a padding token to ignore."""
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.num_attention_heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if key_padding_mask is not None:
            dots = dots.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_transformer_layers,
        num_attention_heads,
        attention_head_dim,
        mlp_hidden_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(
                            embedding_dim,
                            num_attention_heads=num_attention_heads,
                            attention_head_dim=attention_head_dim,
                            dropout=dropout,
                        ),
                        FeedForward(embedding_dim, mlp_hidden_dim, dropout=dropout),
                    ]
                )
                for _ in range(num_transformer_layers)
            ]
        )

    def forward(self, x, key_padding_mask=None):
        for attn, ff in self.layers:
            x = attn(x, key_padding_mask=key_padding_mask) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """1D ViT over patches (B, N, W) made in src/patches: one patch is one token."""

    def __init__(
        self,
        *,
        patch_width,
        num_patches,
        num_classes,
        embedding_dim,
        num_transformer_layers,
        num_attention_heads,
        mlp_hidden_dim,
        attention_head_dim,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_width),
            nn.Linear(patch_width, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            embedding_dim,
            num_transformer_layers,
            num_attention_heads,
            attention_head_dim,
            mlp_hidden_dim,
            dropout,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, patches, patch_lengths=None):
        """(B, N, W) -> (B, 1, num_classes); patch_lengths is accepted for the shared call and unused."""
        x = self.to_patch_embedding(patches)

        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, "d -> b 1 d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")

        x = x + self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, "b * d")

        return self.mlp_head(cls_tokens)
