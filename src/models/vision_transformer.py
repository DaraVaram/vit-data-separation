import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # Project patches
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose to (batch_size, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)  # Prepare for attention (num_patches, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        return attn_output.transpose(0, 1)  # Revert to (batch_size, num_patches, embed_dim)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))  # Attention + Residual + Norm
        x = self.norm2(x + self.mlp(x))   # MLP + Residual + Norm
        return x

class VisionTransformerWithIntermediateOutputs(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64, num_heads=8, mlp_dim=128, num_layers=5, num_classes=10, dropout=0.1):
        super(VisionTransformerWithIntermediateOutputs, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_outputs = []
        x = self.patch_embedding(x)
        batch_size, num_patches, _ = x.shape

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, :num_patches + 1, :]

        for layer in self.encoder_layers:
            x = layer(x)
            intermediate_outputs.append(x)

        cls_token_final = x[:, 0]
        final_output = self.mlp_head(cls_token_final)

        return final_output, intermediate_outputs