import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
#from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
#from data_loaders.get_data import get_dataset_loader


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class ModiffaeTrainLoop:
    def __init__(self, args, train_platform, model, diffusion, train_data, validation_data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        #self.cond_mode = model.cond_mode
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        #self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.semantic_encoder_lr = args.semantic_encoder_lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        #self.use_fp16 = False  # deprecating this option
        #self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        #self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        #self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps

        #print(self.num_steps, len(self.data))
        self.num_epochs = self.num_steps // len(self.train_data) + 1

        #self.num_epochs = 100
        print(f"Number of epochs: {self.num_epochs}")

        #self.sync_cuda = torch.cuda.is_available()

        #print(self.use_fp16)
        #exit()

        self._load_and_sync_parameters()
        #self.mp_trainer = MixedPrecisionTrainer(
        #    model=self.model,
        #    use_fp16=self.use_fp16,
        #    fp16_scale_growth=self.fp16_scale_growth,
        #)

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        #self.opt = AdamW(
        #    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        #)
        #self.opt = AdamW(
        #    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        #)

        #print(self.lr)

        '''self.opt = AdamW(
            [
                {'params': self.model.input_process.parameters()},
                {'params': self.model.sequence_pos_encoder.parameters()},
                {'params': self.model.seqTransEncoder.parameters()},
                {'params': self.model.semantic_encoder.parameters(), 'lr': 1e-5},
                {'params': self.model.embed_timestep.parameters()},
                {'params': self.model.output_process.parameters()}
            ],
            lr=self.lr,
            weight_decay=self.weight_decay
        )'''

        self.opt = AdamW(
            [
                #{'params': self.model.semantic_encoder.parameters(), 'lr': 1e-5},
                {'params': self.model.semantic_encoder.parameters(), 'lr': self.semantic_encoder_lr},
                {'params': self.model.decoder.parameters()}
            ],
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        #print(self.opt.optimizer_specs)
        #exit()

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        # TODO: implement evaluation
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        #if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
        #    mm_num_samples = 0  # mm is super slow hence we won't run it during training
        #    mm_num_repeats = 0  # mm is super slow hence we won't run it during training
        #    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                    split=args.eval_split,
        #                                    hml_mode='eval')

        #    self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                           split=args.eval_split,
        #                                           hml_mode='gt')
        #    self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
        #    self.eval_data = {
        #        'test': lambda: eval_humanml.get_mdm_loader(
        #            model, diffusion, args.eval_batch_size,
        #            gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
        #            args.eval_num_samples, scale=1.,
        #        )
        #    }
        #self.use_ddp = False
        #self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        #print('starting validation')
        #self.args.eval_during_training = True
        #self.evaluate()
        #exit()


        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.train_data): # the condition now includes distances for karate
                
                #distance = cond['distance']
                #print(distance)

                #if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                #    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:

                    if self.validation_data is not None:
                        self.model.eval()
                        self.run_validation()
                        self.model.train()

                    for k, v in logger.get_current().name2val.items():
                        if k == 'train_loss':
                            print('step[{}]: train loss[{:0.5f}]'.format(self.step+self.resume_step, v))
                        if k == 'validation_loss':
                            print('step[{}]: validation loss[{:0.5f}]'.format(self.step + self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            #if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            #    break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        '''if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')'''

        #elif self.dataset in ['humanact12', 'uestc']:
        if self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.train_data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def run_validation(self):
        # iterate over validation batches
        for motion, cond in tqdm(self.validation_data, desc='Validation round'):  # the condition now includes distances for karate

            # if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            #     break

            motion = motion.to(self.device)
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

            self.forward(motion, cond, split='validation')

    def run_step(self, batch, cond):
        loss = self.forward(batch, cond, split='train')
        loss.backward()
        # Anthony: note that by this I removed the logging for
        # gradient and param normalization
        #self.mp_trainer.optimize(self.opt)
        self.opt.step()
        #self._anneal_lr()
        self.log_step()

    def forward(self, batch, cond, split):
        #self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.batch_size):
            # Eliminates the microbatch feature
            assert i == 0
            #assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.batch_size) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                #self.ddp_model,
                self.model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.train_data.dataset
            )

            #if last_batch or not self.use_ddp:
            losses = compute_losses()
            #else:
            #    with self.ddp_model.no_sync():
            #        losses = compute_losses()

            #print(isinstance(self.schedule_sampler, LossAwareSampler))
            if isinstance(self.schedule_sampler, LossAwareSampler) and split == 'train':
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}, split
            )
            #self.mp_trainer.backward(loss)
            #loss.backward()
            return loss

    # Anthony: important not to use this because
    # of the different learning rates in different parts of the model.
    # If I wish to do lr decay this needs to be adjusted.
    '''def _anneal_lr(self):
        print(not self.lr_anneal_steps)
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr'''

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        #logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


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
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses, split):
    for key, values in losses.items():
        logger.logkv_mean(f'{split}_{key}', values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{split}_{key}_q{quartile}", sub_loss)
