import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

if __name__ == "__main__":
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)

# from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset

# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetTimmMod1Policy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetTimmMod1Policy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder["reduce_pretrained_lr"]:
            obs_encorder_lr *= 0.1
            print("==> reduce pretrained obs_encorder's lr")
        pretraiend_obs_encorder_params = list()
        # NOTE: avoid hardcoding value in the future, since we only have 1 pretrained model now, it's fine.
        print("==> rgb keys: ", self.model.obs_encoder.rgb_keys)
        for key in self.model.obs_encoder.rgb_keys:
            for param in self.model.obs_encoder.key_model_map[key].parameters():
                if param.requires_grad:
                    pretraiend_obs_encorder_params.append(param)
        print(f"obs_encorder params: {len(pretraiend_obs_encorder_params)}")
        param_groups = [
            {"params": self.model.model_sparse.parameters()},
            {"params": pretraiend_obs_encorder_params, "lr": obs_encorder_lr},
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")
        self.optimizer = torch.optim.AdamW(params=param_groups, **optimizer_cfg)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ["optimizer"]

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with="wandb")
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        sparse_normalizer_path = os.path.join(self.output_dir, "sparse_normalizer.pkl")
        if accelerator.is_main_process:
            sparse_normalizer = dataset.get_normalizer()
            pickle.dump(sparse_normalizer, open(sparse_normalizer_path, "wb"))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        sparse_normalizer = pickle.load(open(sparse_normalizer_path, "rb"))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print(
            "train dataset:", len(dataset), "train dataloader:", len(train_dataloader)
        )
        print("val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader))

        self.model.set_normalizer(sparse_normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(sparse_normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner, output_dir=self.output_dir
        # )
        # assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader,
                val_dataloader,
                self.model,
                self.optimizer,
                lr_scheduler,
            )
        )

        # print batch size
        batch_size = cfg.dataloader.batch_size
        print(f"batch_size: {batch_size}")
        sample_batch = next(iter(train_dataloader))
        for key, attr in sample_batch["obs"]["sparse"].items():
            print("obs.sparse.key: ", key, attr.shape)
        print("obs.sparse: ", sample_batch["action"]["sparse"].shape)

        # print action dimension
        print("action: ", sample_batch["action"]["sparse"].shape)
        print("dataset.action_type: ", dataset.action_type)
        action_dimension = sample_batch["action"]["sparse"].shape[-1]

        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        args = {}
        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                    disable=False
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )

                        # always use the latest batch
                        # except for the last batch of an epoch (last batch might not have full batch size)
                        if (
                            batch_idx == 0
                            or batch["action"]["sparse"].shape[0] == batch_size
                        ):
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch, args)

                        # loss = raw_loss / cfg.training.gradient_accumulate_every
                        # loss.backward()
                        accelerator.backward(raw_loss)

                        # compute gradient for logging
                        if cfg.training.log_gradient_norm:
                            sparse_grads = [
                                param.grad.detach().flatten()
                                for param in model_unwrapped.model_sparse.parameters()
                                if param.grad is not None
                            ]
                            sparse_norm = (
                                torch.cat(sparse_grads).norm()
                                if len(sparse_grads) > 0
                                else 0
                            )

                        # step optimizer
                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        model_unwrapped = accelerator.unwrap_model(self.model)
                        raw_loss_cpu = raw_loss.item()
                        sparse_loss = model_unwrapped.get_loss_components()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "sparse_loss": sparse_loss,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                def log_action_mse(step_log, category, pred_action, gt_action):
                    pred_naction = {
                        "sparse": sparse_normalizer["action"].normalize(
                            pred_action["sparse"]
                        ),
                    }
                    gt_naction = {
                        "sparse": sparse_normalizer["action"].normalize(
                            gt_action["sparse"]
                        ),
                    }
                    B, T, _ = pred_naction["sparse"].shape
                    pred_naction_sparse = pred_naction["sparse"].view(
                        B, T, -1, action_dimension
                    )
                    gt_naction_sparse = gt_naction["sparse"].view(
                        B, T, -1, action_dimension
                    )
                    step_log[f"{category}_sparse_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse, gt_naction_sparse
                        )
                    )
                    step_log[f"{category}_sparse_cmd_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., :9], gt_naction_sparse[..., :9]
                        )
                    )
                    step_log[f"{category}_sparse_vt_naction_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., 9:18], gt_naction_sparse[..., 9:18]
                        )
                    )
                    step_log[f"{category}_sparse_stiffness_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_naction_sparse[..., 18], gt_naction_sparse[..., 18]
                        )
                    )

                # run diffusion sampling on a training batch
                if (
                    self.epoch % cfg.training.sample_every
                ) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        gt_action = batch["action"]
                        pred_action = policy.predict_action(batch["obs"])

                        log_action_mse(step_log, "train", pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(
                                val_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            gt_action = batch["action"]
                            pred_action = policy.predict_action(batch["obs"])
                            log_action_mse(step_log, "val", pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action

                # checkpoint
                if (
                    self.epoch % cfg.training.checkpoint_every
                ) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
