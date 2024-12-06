import pickle
import numpy as np

import torch
import random
import grid2op
import argparse
import numpy as np

from utils import set_config_for_env, set_config_from_env_eval, StateProcesser
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Reward import MyRedispReward, MyRedispRewardNew, L2RPNReward
# agent to train model
from Agent.DDPGAgent import DDPGAgent


class Grid2OPEnv:
    pass

parser = argparse.ArgumentParser(description='Grid system control program')

parser.add_argument('--env_type', default='rte_case118_example' )
parser.add_argument('--env_fix_start_point', default=0 )
parser.add_argument('--env_start_range', default=4000 )
parser.add_argument('--env_quick_reconnect_line', default=0 )
parser.add_argument('--env_soft_overflow_allow', default=0 )
parser.add_argument('--env_hard_overflow_range', default=2 )
parser.add_argument('--env_ban_overflow', default=0 )
parser.add_argument('--agent_type', default='DDPG' )
parser.add_argument('--buffer_type', default='FIFO' )
parser.add_argument('--max_episode', default=200000 )
parser.add_argument('--max_timestep', default=10000 )
parser.add_argument('--max_buffer_size', default=50000 )
parser.add_argument('--min_state', default=1 )
parser.add_argument('--model_update_freq', default=1024 )
parser.add_argument('--model_save_freq', default=20 )
parser.add_argument('--init_model', default=1 )
parser.add_argument('--save_model', default=0 )
parser.add_argument('--output_res', default=0 )
parser.add_argument('--load_state_normalization', default=0 )
parser.add_argument('--update_state_normalization', default=1 )
parser.add_argument('--use_mini_batch', default=1 )
parser.add_argument('--use_state_norm', default=1 )
parser.add_argument('--reflect_actionspace', default=0 )
parser.add_argument('--add_balance_loss', default=1 )
parser.add_argument('--balance_loss_rate', default=0.01 )
parser.add_argument('--balance_loss_rate_decay_rate', default=0.0 )
parser.add_argument('--balance_loss_rate_decay_freq', default=10000 )
parser.add_argument('--min_balance_loss_rate', default=0.001 )
parser.add_argument('--split_balance_loss', default=1 )
parser.add_argument('--danger_region_rate', default=0.1 )
parser.add_argument('--save_region_rate', default=0.6 )
parser.add_argument('--save_region_balance_loss_rate', default=0.0001 )
parser.add_argument('--warning_region_balance_loss_rate', default=0.001 )
parser.add_argument('--danger_region_balance_loss_rate', default=0.01 )
parser.add_argument('--use_hierarchical_agent', default=1 )
parser.add_argument('--use_history_state', default=1 )
parser.add_argument('--use_history_action', default=0 )
parser.add_argument('--history_state_len', default=25 )
parser.add_argument('--gru_num_layers', default=2 )
parser.add_argument('--gru_hidden_size', default=64 )
parser.add_argument('--use_topology_info', default=0 )
parser.add_argument('--active_function', default='tanh' )
parser.add_argument('--punish_balance_out_range', default=0 )
parser.add_argument('--punish_balance_out_range_rate', default=0.0 )
parser.add_argument('--reward_from_env', default=1 )
parser.add_argument('--reward_for_survive', default=0.0 )
parser.add_argument('--lr_actor', default=1e-5 )
parser.add_argument('--lr_critic', default=1e-3 )
parser.add_argument('--lr_decay_step_size', default=400 )
parser.add_argument('--lr_decay_gamma', default=0.9 )
parser.add_argument('--init_action_std', default=0.01 )
parser.add_argument('--min_action_std', default=0.01 )
parser.add_argument('--batch_size', default=32 )
parser.add_argument('--mini_batch_size', default=32 )
parser.add_argument('--sample_data', default=1 )
parser.add_argument('--sample_data_dir', default="/home/terigen/grid2op_mod/sample_data/2" )
parser.add_argument('--sampel_step_num', default=1000000 )
# parser.add_argument('--load_model', default=1 )
# parser.add_argument('--model_load_path', default="/home/LAB/zhutc/qiuyue/grid2op_mod/save_model/hihdm_model/"\
                    # "train_grid_system_DDPG_useSplitBalanceLoss_useHierarchical_useHisState_hisLen25/200_save_model.pth")


import os
home_dir = os.path.expanduser("~")

def load_data(filename = os.path.join(home_dir, "mod-decision-diffuser/code/diffuser/datasets/data/data_sampled.pkl")):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data['data'], data['info']


def sequence_dataset(env, preprocess_fn, seed: int = None):
    """
    Returns an iterator through trajectories.
    Args:
        env: An MultiAgentEnv object.
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
    data, data_info = load_data()
    last_index = 0
    for terminates in data_info['traj_lens']:
        episode = {}
        episode['observations'] = np.array(data['observations'][last_index:terminates])
        episode['actions'] = np.array(data['actions'][last_index:terminates])
        episode['rewards'] = np.array(data['rewards'][last_index:terminates])
        episode['terminals'] = np.array(data['terminals'][last_index:terminates])
        assert episode['terminals'][-1] == True
        last_index = terminates
        yield episode


def make_env(param, grid_config):
    env = grid2op.make(
        grid_config.env_type, test=True, backend=PandaPowerBackend(),
        reward_class=MyRedispRewardNew if grid_config.env_type == 'l2rpn_wcci_2022_dev' else MyRedispReward,
        param=param
    )
    return env

def load_environment(name, **kwargs):

    env = make_env(param, grid_config)
    return env


grid_config = parser.parse_args()
grid_config.device = "cuda"
# set data to calculate split balance loss
if grid_config.split_balance_loss:
    grid_config.danger_balance_loss_rate = torch.tensor([
        grid_config.danger_region_balance_loss_rate for _ in range(grid_config.mini_batch_size if grid_config.use_mini_batch else grid_config.batch_size)
    ]).to(grid_config.device)
    grid_config.warning_balance_loss_rate = torch.tensor([
        grid_config.warning_region_balance_loss_rate for _ in range(grid_config.mini_batch_size if grid_config.use_mini_batch else grid_config.batch_size)
    ]).to(grid_config.device)
    grid_config.save_balance_loss_rate = torch.tensor([
        grid_config.save_region_balance_loss_rate for _ in range(grid_config.mini_batch_size if grid_config.use_mini_batch else grid_config.batch_size)
    ]).to(grid_config.device)

param = set_config_for_env(grid_config)
null_env = make_env(param, grid_config)
# # set basic info in config
set_config_from_env_eval(grid_config, null_env)
my_agent = DDPGAgent(null_env, grid_config)
state_processer = StateProcesser(grid_config)
