import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.win_size = win_size
        window_size = win_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.distances = False

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        # Sigma 값 간소화 및 최적화된 거리 계산

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        dynamic_distances = self.calculate_dynamic_distances(sigma, B, H, L, queries.device)
        self.distances = dynamic_distances
        gaussian_kernel = self.calculate_dynamic_gaussian_kernel(self.distances, sigma, B, H, L)


        # 최종 attention 가중치 계산
        series = self.dropout(torch.softmax(attn * gaussian_kernel, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, gaussian_kernel, sigma)
        else:
            return (V.contiguous(), None)

    def calculate_dynamic_distances(self, sigma, B, H, L, device):
        indices = torch.arange(L, device=device).repeat(B * H, 1)  # B*H x L
        if self.distances is False :
            print("False")
            self.distances = indices + sigma.view(B * H, L)
        else :
            self.distances = sigma.view(B * H, L) * sigma.view(B * H, L)
        # dynamic_indices = self.distances + sigma.view(B * H, L)  # Apply sigma as dynamic offset
        distances = torch.abs(self.distances.unsqueeze(-1) - self.distances.unsqueeze(-2))
        print("distances", distances)
        return distances.view(B, H, L, L)

    def calculate_dynamic_gaussian_kernel(self, distances, sigma, B, H, L):
        sigma = sigma.view(B, H, L, 1).expand(-1, -1, -1, L)
        gaussian_kernel = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-distances ** 2 / (2 * sigma ** 2))
        return gaussian_kernel



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
