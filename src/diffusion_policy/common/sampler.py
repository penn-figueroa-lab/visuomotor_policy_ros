import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../../"))

from typing import Optional, Callable
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

from PyriteUtility.data_pipeline.indexing import (
    get_sample_ids,
    get_samples,
    get_dense_query_points_in_horizon,
)
from PyriteUtility.data_pipeline.data_plotting import plot_sample


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    """Sample sequences of observations and actions from replay buffer.
    Query ID is based on rgb data, which is likely to be the most sparse data.
    Other data corresponding to the query ID is obtain based on timestamps.
    1. Given query id, find the corresponding low dim/rgb id.
    1. Construct sparse sample:
        Sample sparse obs horizon before idx,
        Sample sparse action horizon after idx.
    2. Construct dense sample:
        Find the indices of dense query points. For each dense query point:
            Sample dense obs horizon before idx,
            Sample dense action horizon after idx.
    """

    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: dict,
        obs_to_obs_sample: Callable,
        action_to_action_sample: Callable,
        id_list: list,
        sparse_query_frequency_down_sample_steps: int = 1,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
    ):
        episode_keys = replay_buffer["data"].keys()
        # Step one: Find the usable length of each episode
        episodes_length = replay_buffer["meta"]["episode_rgb0_len"][:]
        print("episodes_length: ",episodes_length)
        episodes_length_for_query = episodes_length.copy()
        episodes_start = episodes_length.copy()
        if not action_padding:
            # if no action padding, truncate the indices to query so the last query point
            #  still has access to the whole horizon of actions
            #  This is enforced by sparse action alone. It is assumed that the dense action is
            #  not affected.
            sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
            sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
                "down_sample_steps"
            ]
            action_chopped_len = sparse_action_horizon * sparse_action_down_sample_steps
            episode_count = -1
            for episode in episode_keys:
                episode_count += 1

                low_dim_end_time = 1e9
                low_dim_start_time = -1e9
                for id in id_list:
                    end_time = replay_buffer["data"][episode]["obs"][
                        f"robot_time_stamps_{id}"
                    ][-action_chopped_len - 1]
                    low_dim_end_time = min(low_dim_end_time, end_time)

                    start_time = replay_buffer["data"][episode]["obs"][
                        f"robot_time_stamps_{id}"
                    ][0]
                    low_dim_start_time = max(low_dim_start_time, start_time)

                # This is needed when there are multiple cameras
                rgb_end_time = 1e9
                rgb_start_time = -1e9
                for id in id_list:
                    end_time = replay_buffer["data"][episode]["obs"][
                        f"rgb_time_stamps_{id}"
                    ][-1]
                    rgb_end_time = min(rgb_end_time, end_time)

                    start_time = replay_buffer["data"][episode]["obs"][
                        f"rgb_time_stamps_{id}"
                    ][0]
                    rgb_start_time = max(rgb_start_time, start_time)

                end_time = min(low_dim_end_time, rgb_end_time)
                start_time = max(low_dim_start_time, rgb_start_time)

                last_rgb_idx = 1e9
                first_rgb_idx = -1
                for id in id_list:
                    rgb_times = replay_buffer["data"][episode]["obs"][
                        f"rgb_time_stamps_{id}"
                    ]
                    # find the last rgb_times index that is before low_dim_end_time
                    rgb_id = np.searchsorted(rgb_times, end_time, side="right") - 1
                    last_rgb_idx = min(last_rgb_idx, rgb_id)

                    # find the first rgb_times index that is after low_dim_start_time
                    rgb_id = np.searchsorted(rgb_times, start_time, side="left")
                    first_rgb_idx = max(first_rgb_idx, rgb_id)

                episodes_length_for_query[episode_count] = last_rgb_idx - first_rgb_idx
                episodes_start[episode_count] = first_rgb_idx
        print("episodes_length_for_query: ",episodes_length_for_query)
        assert np.min(episodes_length_for_query) > 0

        # Step two: Computes indices from episodes_length_for_query. indices[i] = (epi_id, epi_len, id)
        #   epi_id: which episode the index i belongs to.
        #   epi_len: length of the episode.
        #   id: the index within the episode.
        epi_id = []
        epi_len = []
        ids = []
        episode_count = -1
        for key in episode_keys:
            episode_count += 1
            episode_index = int(key.split("_")[-1])
            array_length = episodes_length_for_query[episode_count]
            if episode_mask is not None and not episode_mask[episode_count]:
                # skip episode
                continue
            epi_id.extend([episode_index] * array_length)
            epi_len.extend([episodes_length[episode_count]] * array_length)
            ids.extend(episodes_start[episode_count] + np.arange(array_length))

        # Step three: Down sample the query indices to make the dataset smaller
        epi_id = epi_id[::sparse_query_frequency_down_sample_steps]
        epi_len = epi_len[::sparse_query_frequency_down_sample_steps]
        ids = ids[::sparse_query_frequency_down_sample_steps]

        indices = list(zip(epi_id, epi_len, ids))

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.obs_to_obs_sample = obs_to_obs_sample
        self.action_to_action_sample = action_to_action_sample
        self.id_list = id_list

        self.ignore_rgb_is_applied = (
            False  # speed up the interation when getting normalizer
        )
        self.flag_has_dense = "dense" in self.shape_meta["sample"]["obs"]

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        """Sample a sequence of observations and actions at idx."""
        epi_id, epi_len_rgb, rgb_id = self.indices[idx]
        episode = f"episode_{epi_id}"
        data_episode = self.replay_buffer["data"][episode]

        # indices are counted for the rgb0 obs data.
        # To get others (rgb, low dim, action), we need to find their id
        query_time = data_episode["obs"]["rgb_time_stamps_0"][rgb_id]
        sparse_obs_unprocessed = dict()
        for key, attr in self.shape_meta["sample"]["obs"]["sparse"].items():
            input_arr = data_episode["obs"][key]
            this_horizon = attr["horizon"]
            this_downsample_steps = attr["down_sample_steps"]
            type = self.shape_meta["obs"][key]["type"]

            if "rgb" in key:
                id = int(key.split("_")[-1])
            else:
                id = int(key[5])  # robot0_xxxx

            # find the query id for the query time
            if "rgb" in key:
                query_id = np.searchsorted(
                    data_episode["obs"][f"rgb_time_stamps_{id}"], query_time
                )
                found_time = data_episode["obs"][f"rgb_time_stamps_{id}"][query_id]

                if abs(found_time - query_time) > 50.0:
                    print("processing key: ", key)
                    print("query_time: ", query_time)
                    print(
                        "total time: ",
                        data_episode["obs"][f"rgb_time_stamps_{id}"][-1],
                    )
                    print("query_id: ", query_id)
                    print(
                        "total id: ",
                        len(data_episode["obs"][f"rgb_time_stamps_{id}"]),
                    )
                    raise ValueError(
                        f"[sampler] {episode} Warning: closest rgb data point at {found_time} is far from the query_time {query_time}"
                    )
            elif "wrench" in key:
                query_id = np.searchsorted(
                    data_episode["obs"][f"wrench_time_stamps_{id}"], query_time
                )
                found_time = data_episode["obs"][f"wrench_time_stamps_{id}"][query_id]
                if abs(found_time - query_time) > 10.0:
                    print("query_time: ", query_time)
                    print(
                        "total time: ",
                        data_episode["obs"][f"wrench_time_stamps_{id}"][-1],
                    )
                    print("query_id: ", query_id)
                    print(
                        "total id: ",
                        len(data_episode["obs"][f"wrench_time_stamps_{id}"]),
                    )
                    raise ValueError(
                        f"[sampler] {episode} Warning: closest wrench data point at {found_time} is far from the query_time {query_time}"
                    )
            else:
                # eef_pos or eef_rot_axis_angle
                query_id = np.searchsorted(
                    data_episode["obs"][f"robot_time_stamps_{id}"], query_time
                )
                found_time = data_episode["obs"][f"robot_time_stamps_{id}"][query_id]
                error = found_time - query_time
                if abs(found_time - query_time) > 10.0:
                    raise ValueError(
                        f"[sampler] {episode} Warning: closest robot data point at {found_time} is far from query_time {query_time}"
                    )

            # how many obs frames before the query time are valid
            num_valid = min(this_horizon, query_id // this_downsample_steps + 1)
            slice_start = query_id - (num_valid - 1) * this_downsample_steps
            assert slice_start >= 0

            # sample every this_downsample_steps frames from slice_start to id+1
            if type == "rgb":
                if self.ignore_rgb_is_applied:
                    continue
                output = input_arr[slice_start : query_id + 1 : this_downsample_steps]
            elif type == "low_dim":
                output = input_arr[
                    slice_start : query_id + 1 : this_downsample_steps
                ].astype(np.float32)
            assert output.shape[0] == num_valid
            # solve padding
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
            sparse_obs_unprocessed[key] = output

        # sparse action
        action_id = np.searchsorted(data_episode["action_time_stamps"], query_time)
        found_time = data_episode["action_time_stamps"][action_id]
        # print("error: ",found_time - query_time)
        if abs(found_time - query_time) > 5.0:
            print("error break: ",found_time - query_time)
            raise ValueError(
                f"[sampler] {episode} Warning: action found_time {found_time} is not equal to query_time {query_time}"
            )
        input_arr = data_episode["action"]
        action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        action_down_sample_steps = self.shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]
        slice_end = min(
            len(input_arr) - 1,
            action_id + (action_horizon - 1) * action_down_sample_steps + 1,
        )
        sparse_action_unprocessed = input_arr[
            action_id:slice_end:action_down_sample_steps
        ].astype(np.float32)
        # solve padding
        if not self.action_padding:
            assert sparse_action_unprocessed.shape[0] == action_horizon
        elif sparse_action_unprocessed.shape[0] < action_horizon:
            padding = np.repeat(
                sparse_action_unprocessed[-1:],
                action_horizon - sparse_action_unprocessed.shape[0],
                axis=0,
            )
            sparse_action_unprocessed = np.concatenate(
                [sparse_action_unprocessed, padding], axis=0
            )

        dense_obs_unprocessed = {}
        dense_action_unprocessed = []
        if self.flag_has_dense:
            sparse_action_horizon = self.shape_meta["sample"]["action"]["sparse"][
                "horizon"
            ]
            sparse_action_down_sample_steps = self.shape_meta["sample"]["action"][
                "sparse"
            ]["down_sample_steps"]

            a_dense_obs_key = next(
                iter(self.shape_meta["sample"]["obs"]["dense"].values())
            )
            dense_obs_horizon = a_dense_obs_key["horizon"]
            dense_obs_down_sample_steps = a_dense_obs_key["down_sample_steps"]
            dense_action_horizon = self.shape_meta["sample"]["action"]["dense"][
                "horizon"
            ]
            dense_action_down_sample_steps = self.shape_meta["sample"]["action"][
                "dense"
            ]["down_sample_steps"]

            # compute local queries based on sparse/dense horizon and down sample steps.
            # Note that the same local ID is used for both obs and action,
            # This is assuming dense obs (low dim) and action has aligned raw data.
            dense_queries_local = get_dense_query_points_in_horizon(
                sparse_action_horizon,
                sparse_action_down_sample_steps,
                dense_action_horizon,
                dense_action_down_sample_steps,
                delta_steps=self.shape_meta["sample"]["action"][
                    "dense_sample_delta_steps"
                ],
            )
            dense_obs_queries = low_dim_id + dense_queries_local
            dense_action_queries = action_id + dense_queries_local

            # dense obs (H, T, D)
            # assuming no padding is needed
            dense_obs_sample_ids = get_sample_ids(
                dense_obs_queries,
                dense_obs_horizon,
                dense_obs_down_sample_steps,
                backwards=True,
                closed=False,
            )
            dense_obs_sample_ids = np.maximum(dense_obs_sample_ids, 0)

            for key in self.shape_meta["sample"]["obs"]["dense"].keys():
                input_arr = data_episode["obs"][key]
                dense_obs_unprocessed[key] = input_arr[dense_obs_sample_ids].astype(
                    np.float32
                )

            # dense action (H, T, D)
            # assuming no padding is needed
            dense_action_unprocessed = get_samples(
                data_episode["action"],
                dense_action_queries,
                dense_action_horizon,
                dense_action_down_sample_steps,
                backwards=False,
                closed=True,
            )

        #   convert to relative pose
        obs_sample = self.obs_to_obs_sample(
            obs_sparse=sparse_obs_unprocessed,
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
            ignore_rgb=self.ignore_rgb_is_applied,
        )
        action_sample = self.action_to_action_sample(
            action_sparse=sparse_action_unprocessed,
            id_list=self.id_list,
        )
        return obs_sample, action_sample

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
