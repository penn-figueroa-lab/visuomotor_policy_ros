import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import copy
from typing import Dict, Optional

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
from einops import rearrange, reduce

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

import sys
import os

from PyriteConfig.tasks.common.common_type_conversions import (
    raw_to_obs,
    raw_to_action9,
    raw_to_action19,
    obs_to_obs_sample,
    action9_to_action_sample,
    action19_to_action_sample,
)

from PyriteUtility.planning_control.trajectory import LinearInterpolator
from PyriteUtility.spatial_math.spatial_utilities import rot6_to_SO3, SO3_to_rot6d
from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()


class VirtualTargetDataset(BaseDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        sparse_query_frequency_down_sample_steps: int = 1,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        normalize_wrench: bool = False,
    ):
        # load into memory store
        print("[VirtualTargetDataset] loading data into store")
        with zarr.DirectoryStore(dataset_path) as directory_store:
            replay_buffer_raw = ReplayBuffer.copy_from_store(
                src_store=directory_store, dest_store=zarr.MemoryStore()
            )
        # convert raw to replay buffer
        print("[VirtualTargetDataset] raw to obs/action conversion")
        action_type = "pose9"  # "pose9" or "pose9pose9s1"
        if shape_meta["action"]["shape"][0] == 9:
            action_type = "pose9"
        elif (
            shape_meta["action"]["shape"][0] == 19
            or shape_meta["action"]["shape"][0] == 38
        ):
            action_type = "pose9pose9s1"
        else:
            raise RuntimeError("unsupported")
        self.action_type = action_type
        self.id_list = shape_meta["id_list"]
        replay_buffer = self.raw_episodes_conversion(replay_buffer_raw, shape_meta)

        # train/val mask for training
        val_mask = get_val_mask(
            n_episodes=replay_buffer_raw.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        print("[VirtualTargetDataset] creating SequenceSampler.")
        if action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        else:
            raise RuntimeError("unsupported")

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=self.id_list,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.sparse_query_frequency_down_sample_steps = (
            sparse_query_frequency_down_sample_steps
        )
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.normalize_wrench = normalize_wrench

    def raw_episodes_conversion(
        self, replay_buffer_raw: ReplayBuffer, shape_meta: dict
    ):
        replay_buffer = dict()
        replay_buffer["data"] = dict()

        for ep in replay_buffer_raw["data"].keys():
            # iterates over episodes
            # ep: 'episode_xx'
            replay_buffer["data"][ep] = dict()
            raw_to_obs(
                replay_buffer_raw["data"][ep], replay_buffer["data"][ep], shape_meta
            )
            if self.action_type == "pose9":
                raw_to_action9(replay_buffer_raw["data"][ep], replay_buffer["data"][ep])
            elif self.action_type == "pose9pose9s1":
                raw_to_action19(
                    replay_buffer_raw["data"][ep],
                    replay_buffer["data"][ep],
                    self.id_list,
                )
            else:
                raise RuntimeError("unsupported")
        # meta
        replay_buffer["meta"] = replay_buffer_raw["meta"]
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if self.action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif self.action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        else:
            raise RuntimeError("unsupported")

        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=self.id_list,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> tuple:
        """Compute normalizer for each key in the dataset.
        Note: only low_dim and action are considered. Image does not need normalization.
        """
        sparse_normalizer = LinearNormalizer()

        # gather all data in cache
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=16,
        )

        data_cache_sparse = {}
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            # sparse obs
            for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
                if self.shape_meta["obs"][key]["type"] == "low_dim":
                    if key not in data_cache_sparse.keys():
                        data_cache_sparse[key] = []
                    data_cache_sparse[key].append(
                        copy.deepcopy(batch["obs"]["sparse"][key])
                    )
            if "action" not in data_cache_sparse.keys():
                data_cache_sparse["action"] = []
            data_cache_sparse["action"].append(copy.deepcopy(batch["action"]["sparse"]))
        self.sampler.ignore_rgb(False)

        # concatenate all data
        for key in data_cache_sparse.keys():
            # data[key] = (# batches, B, T, D)
            data_cache_sparse[key] = np.concatenate(data_cache_sparse[key])  # (B, T, D)
            if not self.temporally_independent_normalization:
                data_cache_sparse[key] = rearrange(
                    data_cache_sparse[key], "B T ... -> (B T) (...)"
                )  # (B*T, D)

        # sparse: compute normalizer for action
        sparse_action_normalizers = list()
        print("data_cache_sparse['action']", data_cache_sparse["action"].shape)
        for i in range(len(self.id_list)):
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * 19 + 0 : i * 19 + 3]
                    )
                )
            )  # pos
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * 19 + 3 : i * 19 + 9]
                    )
                )
            )  # rot
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * 19 + 9 : i * 19 + 12]
                    )
                )
            )  # pos
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * 19 + 12 : i * 19 + 18]
                    )
                )
            )  # rot
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * 19 + 18 : i * 19 + 19]
                    )
                )
            )  # stiffness

        sparse_normalizer["action"] = concatenate_normalizer(sparse_action_normalizers)

        # sparse: compute normalizer for obs
        for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
            type = self.shape_meta["obs"][key]["type"]
            if type == "low_dim":
                stat = array_to_stats(data_cache_sparse[key])
                if "eef_pos" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif "rot_axis_angle" in key:
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                elif "wrench" in key:
                    if self.normalize_wrench:
                        this_normalizer = get_range_normalizer_from_stat(stat)
                    else:
                        this_normalizer = get_identity_normalizer_from_stat(stat)
                else:
                    raise RuntimeError("unsupported")
                sparse_normalizer[key] = this_normalizer
            elif type == "rgb":
                sparse_normalizer[key] = get_image_identity_normalizer()
            elif type == "timestamp":
                continue
            else:
                raise RuntimeError("unsupported")

        return sparse_normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": dict_apply(action_array, torch.from_numpy),
        }
        return torch_data
