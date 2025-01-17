import os
import collections
import numpy as np
import gym
import pdb
import pickle

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    # env.max_episode_steps = wrapped_env._max_episode_steps
    env.max_episode_steps = 13000
    env.name = name
    return env

def get_dataset(env):
    """
        加载本地数据集
        Args:
            env: 环境对象，可能用于特定的处理
            pkl_file_path: 数据集的本地文件路径
        """
    with open('../diffuser/datasets/data/data_sampled.pkl', 'rb') as pkl_file:
        dataset = pickle.load(pkl_file)

    # index = dataset['data']['terminals'].index(True)
    # print(f"The first True is at index {index}")
    # print(dataset['data'].keys())
    # exit(0)
    if 'antmaze' in str(env).lower():
        # 对于 antmaze 环境进行修正
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    # subset_data = {}
    # limit = 11  # 可调整为适合的数据大小限制
    #
    # for key, value in dataset['data'].items():
    #     if isinstance(value, np.ndarray):  # 如果是 numpy 数组
    #         subset_data[key] = value[:limit]
    #     elif isinstance(value, list):  # 如果是列表
    #         subset_data[key] = value[:limit]
    #
    # subset_data['terminals'][10] = True
    # print(len(subset_data['rewards']))
    return dataset['data']

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    if isinstance(dataset['rewards'], list):
        dataset['rewards'] = np.array(dataset['rewards'])

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data 
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
