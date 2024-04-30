import torch
import torch.nn as nn
import torch.nn.functional as F

class GELUActivation(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class BlipAttention(nn.Module):
    def __init__(self, embed_dim):
        super(BlipAttention, self).__init__()
        self.dropout = nn.Dropout(p=0.0)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, v)
        x = x.permute(1, 2, 0, 3).reshape(B, N, C)
        x = self.projection(x)
        return x

class BlipMLP(nn.Module):
    def __init__(self, embed_dim):
        super(BlipMLP, self).__init__()
        self.activation_fn = GELUActivation()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

class BlipEncoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super(BlipEncoderLayer, self).__init__()
        self.self_attn = BlipAttention(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mlp = BlipMLP(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_res = x
        x = self.self_attn(x)
        x = x_res + x
        x = self.layer_norm1(x)
        x_res = x
        x = self.mlp(x)
        x = x_res + x
        x = self.layer_norm2(x)
        return x


class BlipDecoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super(BlipDecoderLayer, self).__init__()
        self.self_attn = BlipAttention(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = BlipAttention(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = BlipMLP(embed_dim)

    def forward(self, x, encoder_output):
        x_res = x
        x = self.self_attn(x)
        x = x_res + x
        x = self.layer_norm1(x)

        # Cross-attention
        x_res = x
        x = self.cross_attn(x, encoder_output)
        x = x_res + x
        x = self.layer_norm2(x)

        # MLP
        x_res = x
        x = self.mlp(x)
        x = x_res + x

        return x

class BlipVisionModel(nn.Module):
    def __init__(self, embed_dim=768, num_layers=12):
        super(BlipVisionModel, self).__init__()
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=(16, 16), stride=(16, 16))
        self.encoder = nn.ModuleList([BlipEncoderLayer(embed_dim) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([BlipDecoderLayer(embed_dim) for _ in range(num_layers)])
        self.post_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        for layer in self.encoder:
            x = layer(x)

        for layer in self.decoder:
            x = layer(x)

        x = x.transpose(1, 2).reshape(B, H // 16, W // 16, C)
        x = x.permute(0, 3, 1, 2)
        x = self.post_layernorm(x)
        return x

modelCustom = BlipVisionModel()
print(modelCustom)
