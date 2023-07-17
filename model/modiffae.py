import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import clip
from model.rotation2xyz import Rotation2xyz


class MoDiffAE(nn.Module):
    def __init__(self, num_joints, num_feats, num_frames, translation, pose_rep,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='karate', **kwargs):
        super().__init__()

        #self.legacy = legacy
        #self.modeltype = modeltype
        #self.njoints = njoints
        #self.nfeats = nfeats
        #self.num_actions = num_actions
        ##self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        #self.glob = glob
        #self.glob_rot = glob_rot
        self.translation = translation

        ##self.latent_dim = latent_dim
        # self.semantic_dim = semantic_dim
        #self.num_frames = num_frames

        ##self.ff_size = ff_size
        ##self.num_layers = num_layers
        ##self.num_heads = num_heads
        ##self.dropout = dropout

        #self.ablation = ablation
        ##self.activation = activation
        #self.clip_dim = clip_dim
        #self.action_emb = kargs.get('action_emb', None)

        ##self.input_feats = njoints * nfeats

        self.normalize_output = kwargs.get('normalize_encoder_output', False)

        self.cond_mode = kwargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kwargs.get('cond_mask_prob', 0.)
        #self.arch = arch
        #self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        self.semantic_encoder = SemanticEncoder(
            data_rep=data_rep,
            input_feats=num_joints * num_feats,
            num_frames=num_frames,
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            activation=activation,
            num_layers=num_layers
        )

        self.decoder = Decoder(num_joints, num_feats, num_frames,  #translation, #pose_rep,
                               latent_dim, ff_size, num_layers, num_heads, dropout,
                               activation, data_rep, dataset, **kwargs)

        #self.rot2xyz = Rotation2xyz(device='cpu') #, dataset=self.dataset)
        self.rot2xyz = Rotation2xyz(device='cpu')  # , dataset=self.dataset)

    def forward(self, x, timesteps, y=None):

        #print(y.keys(), 'hi')


        og_motion = y['original_motion']
        # For the manipulation, the new semantic embedding
        # is passed in y.
        if 'semantic_emb' in y.keys():
            #print('using new emb')
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
    def __init__(self, njoints, nfeats, num_frames, #translation, #pose_rep,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='amass', **kargs):
        super().__init__()

        self.n_joints = njoints
        self.nfeats = nfeats
        #self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        #self.pose_rep = pose_rep
        #self.translation = translation

        self.latent_dim = latent_dim
        self.num_frames = num_frames

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.input_feats = self.n_joints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        # TODO: the diffusion autoencoder does not use this and currently during training the mask
        #       probability is ste to 0. Experiment with this and compare results.
        #       This would enable unconditional generation. However, I can simply generate a
        #       condition with another ddim. So the question would more be what is better.
        #       This might negatively effect the training of the semantic encoder.
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Other considerations:
        # - GRU (not as good as transformers according to literature I think)
        # - Transformer decoder: encoder is better suited for this task because ... (see BERT)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                             nhead=self.num_heads,
                                                             dim_feedforward=self.ff_size,
                                                             dropout=self.dropout,
                                                             activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.n_joints,
                                            self.nfeats)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, semantic_emb, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)

        emb += self.mask_cond(semantic_emb, force_mask=force_mask)

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

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
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        #if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        #print(x.shape)
        #print(self.input_feats)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
        #elif self.data_rep == 'rot_vel':
        #    first_pose = x[[0]]  # [1, bs, 150]
        #    first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #    vel = x[1:]  # [seqlen-1, bs, 150]
        #    vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #    return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        #else:
        #    raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape

        #if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        #elif self.data_rep == 'rot_vel':
        #    first_pose = output[[0]]  # [1, bs, d]
        #    first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #    vel = output[1:]  # [seqlen-1, bs, d]
        #    vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #    output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        #else:
        #    raise ValueError

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
    def __init__(self, data_rep, input_feats, num_frames, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.num_frames = num_frames

        self.data_rep = data_rep
        self.input_feats = input_feats

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                             nhead=self.num_heads,
                                                             dim_feedforward=self.ff_size,
                                                             dropout=self.dropout,
                                                             activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=self.num_layers)

        self.linear_time = nn.Linear(
            in_features=self.num_frames,
            out_features=1
        )

    def forward(self, x):

        x = self.input_process(x)

        x_seq = self.sequence_pos_encoder(x)  # [seqlen, bs, d]

        encoder_output = self.seqTransEncoder(x_seq)   # [seqlen, bs, d]

        output = encoder_output.transpose(2, 0)   # # [semdim, bs, seqlen]

        output = self.linear_time(output).squeeze().transpose(1, 0)   # [bs, semdim]

        return output

