import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class SemanticGeneratorTrainLoop:
    def __init__(self, args, train_platform, model, diffusion, semantic_encoder, train_data, validation_data):


        self.args = args
        #self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.semantic_encoder = semantic_encoder
        #self.diffusion = diffusion
        #self.cond_mode = model.cond_mode
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        #self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr

        #print(self.lr)
        # TODO: test even lower lr because train loss jumps to fast in the beginning
        self.lr = args.lr #0.0001  # 005
        #print(self.lr)

        self.log_interval = args.log_interval
        self.save_interval = args.save_interval  # 10_000 #args.save_interval
        #self.resume_checkpoint = args.resume_checkpoint
        #self.use_fp16 = False  # deprecating this option
        #self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        #self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        #self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps

        #print(self.num_steps, len(self.data))
        #self.num_epochs = args.num_epochs #self.num_steps // len(self.train_data) + 1
        self.num_epochs = self.num_steps // len(self.train_data) + 1

        #self.num_epochs = 100
        print(f"Number of epochs: {self.num_epochs}")
        #self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        # lr=0.005
        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())


        #self.sigmoid_fn = torch.nn.Sigmoid()
        #self.loss_fn = torch.nn.BCELoss()

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        # TODO implement evaluation
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.train_data):

                #print(motion.shape)
                #print(cond.shape)

                #out = self.model(motion)

                #exit()

                #while True:
                #    pass

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                             cond['y'].items()}


                #cond = cond.to(self.device)

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:

                    if self.validation_data is not None:
                        self.model.eval()
                        with torch.no_grad():
                            self.run_validation()
                        self.model.train()

                    for k, v in logger.get_current().name2val.items():
                        if k == 'train_loss':
                            print('step[{}]: train loss[{:0.5f}]'.format(self.step, v))
                        if k == 'validation_loss':
                            print('step[{}]: validation loss[{:0.5f}]'.format(self.step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_validation(self):
        for motion, cond in tqdm(self.validation_data, desc='Validation round'):
            motion = motion.to(self.device)
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                         cond['y'].items()}
            self.forward(motion, cond, split='validation')

    def run_step(self, motion, cond):
        loss = self.forward(motion, cond, split='train')
        loss.backward()
        self.opt.step()
        # self._anneal_lr()
        self.log_step()

    def forward(self, motion, cond, split):
        self.opt.zero_grad()

        #print(cond["y"]["labels"].shape[0])
        #exit()

        #t, weights = self.schedule_sampler.sample(cond["y"]["labels"].shape[0], dist_util.dev())
        t, weights = self.schedule_sampler.sample(motion.shape[0], dist_util.dev())

        #t = torch.unsqueeze(t, 1)

        #self.model = self.model.to(self.device)

        with torch.no_grad():
        # I think this needs to be the original motion and not x0
            z_0 = self.semantic_encoder(motion)

        loss = self.diffusion.generator_training_loss(self.model, z_0=z_0, t=t, model_kwargs=cond)
        loss = (loss * weights).mean()

        log_loss_dict(
            {
                'mse': loss
            },
            split
        )

        return loss

    '''def _anneal_lr(self):
        print(not self.lr_anneal_steps)
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr'''

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"model{self.step:09d}.pt"

    def save(self):
        #def save_checkpoint(params):
        def save_checkpoint(state_dict):
            #state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        #save_checkpoint(self.mp_trainer.master_params)
        save_checkpoint(self.model.state_dict())

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.step:09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(losses, split):
    for key, values in losses.items():
        logger.logkv_mean(f'{split}_{key}', values)
