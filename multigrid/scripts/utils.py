from __future__ import annotations

import os
from datetime import datetime

import numpy as np 
from pathlib import Path
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from typing import Callable

def init_globvar():
    global MAX_STEPS, HIDE_STEPS, SEEK_STEPS
    global H, GAMMA, BATCH_SIZE, LR, REPLAY_MEM_SIZE, MIN_REPLAY_MEM_SIZE
    global UPDATE_TARGET_EVERY, EPSILON_DECAY, MIN_EPSILON, DOUBLE_NETWORK 

def preprocess_agent_observations(obs, timestep):
    preprocessed_obs = {}
    for agent in obs: 
        grid_state = obs[agent]['image']
        extra_state_info = [obs[agent]['direction'], 
                            obs[agent]['seeker'], 
                            obs[agent]['curr_pos'], 
                            obs[agent]['other_pos'], 
                            timestep]
        extra_state_info = [np.array(x, ndmin=1) for x in extra_state_info]
        extra_state_info = np.concatenate(extra_state_info)
        preprocessed_obs[agent] = (grid_state, extra_state_info)
    
    return preprocessed_obs

def can_use_gpu() -> bool:
    """
    Return whether or not GPU training is available.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False

def find_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Recursively search for RLlib checkpoints within the given directory.

    If more than one is found, returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Path or str
        The directory to search for checkpoints within
    """
    try:
        checkpoints = Path(search_dir).expanduser().glob('**/*.is_checkpoint')
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent
    except:
        return None

def get_policy_mapping_fn(
    checkpoint_dir: Path | str | None, num_agents: int) -> Callable:
    """
    Create policy mapping function from saved policies in checkpoint directory.
    Maps agent i to the (i % num_policies)-th policy.

    Parameters
    ----------
    checkpoint_dir : Path or str
        The checkpoint directory to load policies from
    num_agents : int
        The number of agents in the environment
    """
    try:
        policies = sorted([
            path for path in (checkpoint_dir / 'policies').iterdir() if path.is_dir()])

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return policies[agent_id % len(policies)].name

        print('Loading policies from:', checkpoint_dir)
        for agent_id in range(num_agents):
            print('Agent ID:', agent_id, 'Policy ID:', policy_mapping_fn(agent_id))

        return policy_mapping_fn

    except:
        return lambda agent_id, *args, **kwargs: f'policy_{agent_id}'

def generate_dir_tree(root_folder, model_name):
    if not os.path.isdir(root_folder):
        os.makedirs(root_folder)
    dateTimeObj = datetime.now()
    dir_name = dateTimeObj.strftime("%m_%d_%H_%M_%S")
    if model_name:
        dir_name = dir_name + "_" + model_name
    dir_name = os.path.join(root_folder, dir_name)
    os.makedirs(dir_name)
    return dir_name