import torch
import torch.nn as nn
from utils import dist_util


class SemanticGenerator(nn.Module):
    def __init__(
            self,
            attribute_dim,
            input_dim,
            latent_dim,
            output_dim,
            condition_dim,
            num_layers,
            dropout):

        super().__init__()

        self.attribute_emb = nn.Linear(attribute_dim, latent_dim)
        self.input_emb = nn.Linear(input_dim, latent_dim)
        self.output_emb = nn.Linear(latent_dim, output_dim)
        self.timestep_emb = TimestepEmbedder(condition_dim)
        self.layers = [AdaLNBlock(latent_dim, condition_dim, dropout) for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, z_t, t, y):
        att = y["labels"]
        att_emb = self.attribute_emb(att)

        ts_emb = self.timestep_emb(t)

        cond = att_emb + ts_emb

        out = self.input_emb(z_t)
        for layer in self.layers:
            out = layer(out, cond)

        out = self.output_emb(out)
        return out


class AdaLNBlock(nn.Module):
    def __init__(
            self,
            latent_dim,
            condition_dim,
            dropout):

        super().__init__()

        self.in_layers = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.cond_layers = nn.Sequential(
            nn.GELU(),
            nn.Linear(condition_dim, 2 * latent_dim),
        )

        self.out_layers = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, cond):
        h = self.in_layers(x)
        cond_out = self.cond_layers(cond)
        scale, shift = torch.chunk(cond_out, 2, dim=1)
        h = h * (1 + scale) + shift
        h = self.out_layers(h)
        return x + h


class TimestepEmbedder(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.emb_dim = condition_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(self, timesteps):
        timesteps = torch.unsqueeze(timesteps, 1)
        timesteps = timesteps.float()
        return self.time_embed(timesteps)
