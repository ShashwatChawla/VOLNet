import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    '''
    Patch Embedding class to convert image into patches and then into embeddings
    '''
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d with patch size as kernel, and stride as patch size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (batch_size, 3, img_size, img_size) -> (batch_size, embed_dim, n_patches ** 2)
        x = self.patch_embed(x)
        x = x.flatten(2)  # flatten to (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, n_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        n_tokens = n_patches + 1  # Add 1 for the class token
        position = torch.arange(0, n_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(n_tokens, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))
        
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # Self-attention in the decoder
        self_attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.norm3(x + self.dropout(mlp_output))
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, out_dim=1024, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, self.patch_embedding.n_patches, dropout)
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CLS token
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize input
        x = self.patch_embedding(x)
        
        # Add the CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand for batch size
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Classification head
        cls_output = x[:, 0]  # Take the CLS token output
        out = self.mlp_head(cls_output)
        return out


if __name__ == "__main__":

    vit = VisionTransformer(
        img_size=640,
        patch_size=16,
        in_channels=14,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        out_dim=1000,
        dropout=0.1
    )

    dummy_data = torch.randn(2, 14, 640, 640)
    output = vit(dummy_data)
    print(f"Output shape: {output.shape}")  # Should output (8, 1000)
