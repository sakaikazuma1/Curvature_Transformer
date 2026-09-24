from torch import nn

from src.architectures.patchtst.backbone import Flatten_Head, TSTiEncoder


class PatchTST(nn.Module):
    """PatchTST over patches (B, N, W) made in src/patches: one patch is one token."""

    def __init__(
        self,
        *,
        patch_width,
        num_patches,
        num_classes,
        embedding_dim=64,
        num_transformer_layers=4,
        num_attention_heads=4,
        mlp_hidden_dim=512,
        attention_head_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = TSTiEncoder(
            1,
            patch_num=num_patches,
            patch_len=patch_width,
            n_layers=num_transformer_layers,
            d_model=embedding_dim,
            n_heads=num_attention_heads,
            d_k=attention_head_dim,
            d_v=attention_head_dim,
            d_ff=mlp_hidden_dim,
            attn_dropout=dropout,
            dropout=dropout,
        )
        self.head = Flatten_Head(False, 1, embedding_dim * num_patches, num_classes)

    def forward(self, patches, patch_lengths=None):
        """(B, N, W) -> (B, num_classes); patch_lengths is accepted for the shared call and unused."""
        # (B, N, W) -> (B, 1, W, N), the (bs, nvars, patch_len, patch_num) layout of TSTiEncoder.
        encoded = self.encoder(patches.unsqueeze(1).permute(0, 1, 3, 2))
        return self.head(encoded)
