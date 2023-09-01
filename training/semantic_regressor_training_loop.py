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

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class SemanticRegressorTrainLoop:
    def __init__(self, args, train_platform, model, train_data, validation_data):


        self.args = args
        #self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        #self.diffusion = diffusion
        #self.cond_mode = model.cond_mode
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        #self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr

        print(self.lr)
        # TODO: manage default parameters for each model in utility
        self.lr = 0.005
        print(self.lr)

        self.log_interval = args.log_interval
        self.save_interval = 10_000 #args.save_interval
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
            self.model.regressor.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        #self.sigmoid_fn = torch.nn.Sigmoid()
        #self.loss_fn = torch.nn.BCELoss()



    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.train_data):

                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                             cond['y'].items()}

                self.run_step(cond)
                if self.step % self.log_interval == 0:

                    if self.validation_data is not None:
                        self.model.regressor.eval()
                        self.run_validation()
                        self.model.regressor.train()

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
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                         cond['y'].items()}
            self.forward(cond, split='validation')

    def run_step(self, cond):
        loss = self.forward(cond, split='train')
        loss.backward()
        self.opt.step()
        # self._anneal_lr()
        self.log_step()

    def forward(self, cond, split):
        self.opt.zero_grad()

        og_motion = cond['y']['original_motion']
        target = cond['y']['labels'].squeeze()

        output = self.model(og_motion)

        #loss = self.loss_fn(self.sigmoid_fn(output), target)

        loss = F.binary_cross_entropy_with_logits(output, target)

        action_output = output[:, :5]
        action_output = F.softmax(action_output)
        skill_level_output = output[:, 5]
        skill_level_output = torch.sigmoid(skill_level_output)
        #skill_level_output = output[:, 5:]
        #skill_level_output = F.softmax(skill_level_output)

        action_target = target[:, :5]
        skill_level_target = target[:, 5]
        #skill_level_target = target[:, 5:]

        action_classifications = torch.argmax(action_output, dim=-1)
        action_labels_idxs = torch.argmax(action_target, dim=-1)
        action_correct_predictions = sum(action_classifications == action_labels_idxs).item()
        acc_technique = action_correct_predictions / len(og_motion)
        #action_total_correct += action_correct_predictions

        '''skill_level_classifications = torch.argmax(skill_level_output, dim=-1)
        skill_level_labels_idxs = torch.argmax(skill_level_target, dim=-1)
        skill_level_correct_predictions = sum(skill_level_classifications == skill_level_labels_idxs).item()
        acc_skill_level = skill_level_correct_predictions / len(og_motion)'''

        mae_skill_level = F.l1_loss(skill_level_target, skill_level_output)
        #skill_level_total_correct += skill_level_correct_predictions

        #total_instances += len(og_motion)



        #print(loss)
        #exit()

        log_loss_dict(
            {
                'bce_w_logits': loss.item(),
                #'bce_loss': loss.item(),
                'acc_technique': acc_technique,
                #'acc_skill_level': acc_skill_level
                'mae_skill_level': mae_skill_level
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
