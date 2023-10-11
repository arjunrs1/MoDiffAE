import torch
import torch.nn as nn


class SemanticGenerator(nn.Module):
    def __init__(
            self,
            semantic_encoder,
            diffusion,
            attribute_dim,
            input_dim,
            latent_dim,
            output_dim,
            condition_dim,
            num_layers,
            dropout):

        super().__init__()

        self.semantic_encoder = semantic_encoder
        self.diffusion = diffusion
        self.attribute_emb = nn.Linear(attribute_dim, latent_dim)
        self.input_emb = nn.Linear(input_dim, latent_dim)
        self.output_emb = nn.Linear(latent_dim, output_dim)
        self.timestep_emb = TimestepEmbedder(condition_dim)
        self.layers = [AdaLNBlock(latent_dim, condition_dim, dropout) for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x_0, t, y):
        with torch.no_grad():
            z_0 = self.semantic_encoder(x_0)

        noise = torch.randn_like(z_0)
        z_t = self.diffusion.q_sample(z_0, t, noise=noise)

        t = t.type("torch.cuda.FloatTensor")
        time_steps = self.diffusion._scale_timesteps(t)

        att = y["labels"]

        #print(att.shape)
        #exit()

        att_emb = self.attribute_emb(att)
        ts_emb = self.timestep_emb(time_steps)
        cond = att_emb + ts_emb


        #print(cond.shape)
        #exit()

        out = self.input_emb(z_t)

        for layer in self.layers:
            out = layer(out, cond)

        out = self.output_emb(out)
        return out, z_0


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
        cond_out = self.cond_layers(cond)  # .type(h.dtype)
        #print(cond_out.shape)
        #exit()
        scale, shift = torch.chunk(cond_out, 2, dim=1)
        #print(scale.shape, shift.shape, h.shape)
        #exit()
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
        return self.time_embed(timesteps)  # .permute(1, 0, 2)
