import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ResnetBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.num_channels = 1024
        self.backbone = resnet50(pretrained=pretrained)

    def forward(self, img):
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        return x


class FeedForwardComponent(nn.Module):

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class CrossAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout=0.1):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, img, text, text_mask=None):
        img = img.transpose(0, 1)
        text = text.transpose(0, 1)

        _img, _img_attention = self.self_attention(img, text, text, key_padding_mask=text_mask)
        img = self.self_attn_layer_norm(img + self.dropout(_img))

        _img = self.feed_forward(img)
        img = self.ff_layer_norm(img + self.dropout(_img))

        return img.transpose(0, 1), _img_attention
