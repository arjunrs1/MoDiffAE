import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import clip
from model.rotation2xyz import Rotation2xyz


class MoDiffAE(nn.Module):
    def __init__(self, num_joints, num_feats, num_frames, pose_rep, translation,
                 modiffae_latent_dim, transformer_feedforward_dim, num_layers, num_heads, dropout,
                 semantic_pool_type, dataset):
        super().__init__()

        self.dataset = dataset
        self.pose_rep = pose_rep
        self.translation = translation

        self.semantic_encoder = SemanticEncoder(
            pose_rep=pose_rep,
            input_feats=num_joints * num_feats,
            num_frames=num_frames,
            latent_dim=modiffae_latent_dim,
            num_heads=num_heads,
            transformer_feedforward_dim=transformer_feedforward_dim,
            dropout=dropout,
            num_layers=num_layers,
            semantic_pool_type=semantic_pool_type
        )

        self.decoder = Decoder(
            num_joints=num_joints,
            num_feats=num_feats,
            num_frames=num_frames,
            pose_rep=pose_rep,
            latent_dim=modiffae_latent_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # TODO: remove device parameter and cpu stuff. This is automatically
        #  executed on the device of the tensor anyways.
        self.rot2xyz = Rotation2xyz(device='cpu')

    def forward(self, x, timesteps, y=None):
        og_motion = y['original_motion']
        # For the manipulation, the new semantic embedding
        # is passed in y.
        if 'semantic_emb' in y.keys():
            semantic_emb = y['semantic_emb']
        else:
            semantic_emb = self.semantic_encoder(og_motion)
        output = self.decoder.forward(x, semantic_emb, timesteps, y)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if self.dataset == 'humanact12':
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.dataset == 'humanact12':
            print('training smpl')
            self.rot2xyz.smpl_model.train(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, num_joints, num_feats, num_frames, pose_rep,
                 latent_dim, transformer_feedforward_dim, num_layers, num_heads, dropout):
        super().__init__()

        self.num_joints = num_joints
        self.num_feats = num_feats
        self.pose_rep = pose_rep

        self.latent_dim = latent_dim
        self.num_frames = num_frames

        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_feats = self.num_joints * self.num_feats
        self.input_process = InputProcess(self.pose_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                             nhead=self.num_heads,
                                                             dim_feedforward=self.transformer_feedforward_dim,
                                                             dropout=self.dropout,
                                                             activation="gelu")

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.output_process = OutputProcess(self.pose_rep, self.input_feats, self.latent_dim, self.num_joints,
                                            self.num_feats)

    def forward(self, x, semantic_emb, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        emb += semantic_emb

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

        output = self.seqTransEncoder(xseq)[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, pose_rep, input_feats, latent_dim):
        super().__init__()
        self.pose_rep = pose_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.pose_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, pose_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.pose_rep = pose_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.pose_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class SemanticEncoder(nn.Module):
    def __init__(self, pose_rep, input_feats, num_frames, latent_dim, transformer_feedforward_dim, num_layers,
                 num_heads, dropout, semantic_pool_type):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_frames = num_frames

        self.pose_rep = pose_rep
        self.input_feats = input_feats
        self.semantic_pool_type = semantic_pool_type

        self.input_process = InputProcess(self.pose_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                             nhead=self.num_heads,
                                                             dim_feedforward=self.transformer_feedforward_dim,
                                                             dropout=self.dropout,
                                                             activation="gelu")

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=self.num_layers)

        if self.semantic_pool_type == 'linear_time_layer':
            self.linear_time = nn.Linear(
                in_features=self.num_frames,
                out_features=1
            )

    def forward(self, x):

        x = self.input_process(x)

        x_seq = self.sequence_pos_encoder(x)  # [seqlen, bs, d]

        encoder_output = self.seqTransEncoder(x_seq)   # [seqlen, bs, d]

        output = encoder_output.transpose(2, 0)   # # [semdim, bs, seqlen]

        if self.semantic_pool_type == 'global_avg_pool':
            output = torch.mean(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'global_max_pool':
            output = torch.amax(output, dim=-1).transpose(1, 0)
        elif self.semantic_pool_type == 'linear_time_layer':
            output = self.linear_time(output).squeeze().transpose(1, 0)
        elif self.semantic_pool_type == 'gated_multi_head_attention_pooling':
            # This could be interesting
            raise Exception("Not implemented.")
        else:
            raise Exception("Pool type not implemented.")

        return output
