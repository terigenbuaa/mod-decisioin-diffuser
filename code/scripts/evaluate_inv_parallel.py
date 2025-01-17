import diffuser.utils as utils
from ml_logger import logger
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from utils import set_seed, set_config_for_env

from diffuser.datasets.grid import make_env, state_processer, grid_config, my_agent
from diffuser.datasets.data_encoder_decoder import load_encode_model, encode_data

from diffuser.datasets.grid import grid_config, null_env


def process_action_space(obs):
    # get action limit
    action_low = [-obs.gen_margin_down]
    action_high = [obs.gen_margin_up]
    if grid_config.storage_available:
        action_low.append(-grid_config.storage_max_p_prod)
        action_high.append(grid_config.storage_max_p_absorb)
    # concatenate action limit
    action_low = np.concatenate(action_low, axis=0)
    action_high = np.concatenate(action_high, axis=0)

    return action_low, action_high

def process_action(obs, action, action_low, action_high, env):
        # set balance redispatch to 0
        action[:grid_config.n_gen][grid_config.name_gen == 'balance'] = 0.
        # clip action to action space
        clip_action = np.clip(action, action_low, action_high).reshape(-1)
        # split action by fixed position
        adjust_gen_p = clip_action[grid_config.detail_action_dim[0][0]: grid_config.detail_action_dim[0][1]]
        adjust_storage_p = clip_action[grid_config.detail_action_dim[1][0]: grid_config.detail_action_dim[1][1]]
        # set action to regular format
        # adjust_gen_p[abs(adjust_gen_p) < 0.1] = 0
        adjust_gen_p[obs.gen_p == 0] = 0
        interact_action = env.action_space({})
        interact_action.redispatch = adjust_gen_p
        # try to reconnect all line that is disconnected
        reconnect_line = np.argwhere((~obs.line_status) & (obs.time_before_cooldown_line == 0)).reshape(-1)
        line_set_status = []
        for line_id in reconnect_line:
            line_set_status.append((line_id, 1))
        # if len(reconnect_line) > 0:
        #     interact_action.line_set_status = line_set_status
        interact_action.line_set_status = line_set_status
        if grid_config.storage_available:
            interact_action.storage_p = adjust_storage_p

        # print('interact_action', interact_action)
        # print('clip_action', clip_action)
        
        return interact_action, clip_action

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    # home_dir = os.path.expanduser("~")
    # loadpath = 'grid-diffuser/weights/grid/default_inv/encode64/predict_epsilon_200_1000000.0/dropout_0.25/sdata/100/checkpoint_encode64/' # '/home/LAB/terigen/grid-diffuser-tmp-sync/weights/grid/default_inv/encode/predict_epsilon_200_1000000.0/dropout_0.25/sdata/100/checkpoint/'
 
        
    logger.print(f"load_step: {Config.load_step}")
    # if Config.save_checkpoints:
    #     loadpath = os.path.join(loadpath, f'state_{Config.load_step}.pt')
    # else:
    #     loadpath = os.path.join(loadpath, 'state.pt')

    # encoded_dim = 64

    # step = 130000
    # model_path = os.path.join(Config.bucket, logger.prefix, f'checkpoint_encode64/state_{step}.pt')
    # encode_model_path = os.path.join(Config.bucket, logger.prefix, f'checkpoint_encode64/encode_model_epoch_{step}_dim_64.pth')

    # encode_model = load_encode_model(encoded_dim, encode_model_path)
    # encode_model.eval()
    
   
    # state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    dataset = dataset_config()
    # renderer = render_config()

    if Config.load_pre_encoder:
        observation_dim = Config.encoded_dim
    else:
        observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        load_pre_encoder=Config.load_pre_encoder,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    # trainer = trainer_config(diffusion, dataset, renderer)
    trainer = trainer_config(diffusion, dataset, None)
    logger.print(utils.report_parameters(model), color='green')
    # trainer.step = state_dict['step']
    # trainer.model.load_state_dict(state_dict['model'])
    # trainer.ema_model.load_state_dict(state_dict['ema'])
    trainer.load(Config.load_step)

    num_eval = 1
    device = Config.device
    episode_rewards = []
    rounds = []

    # encode_model checkpoint
    # checkpoint_path = '/home/LAB/terigen/grid2op_mod/grid2op/MakeEnv/checkpoint_epoch_50_encoded_dim_64.pth'

    for i in range(20):
        episode_reward = 0
        param = set_config_for_env(grid_config)
        env = make_env(param, grid_config)

        assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
        returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

        t = 0
        env.set_id(np.random.randint(0, grid_config.total_chronics, 1)[0])
        obs = env.reset()
        env.fast_forward_chronics(np.random.randint(0, grid_config.env_start_range, 1)[0])

        for timestep in range(grid_config.max_timestep):
    
            env_obs = env.get_obs()
            origin_obs, _ = state_processer(env_obs)
            # Reshape the observation from [409] to [1, 409]
            origin_obs = origin_obs.reshape(1, -1)
            # origin_obs = dataset.normalizer.normalize(torch.Tensor.cpu(origin_obs), 'observations')
            origin_obs = dataset.normalizer.normalize(origin_obs, 'observations')

            if Config.load_pre_encoder:
                obs = encode_data(trainer.model.encode_model, to_torch(origin_obs), eval=True)
            else:
                obs = origin_obs
            # import pdb; pdb.set_trace()
            
            conditions = {0: to_torch(obs, device=device)}
            samples = trainer.ema_model.conditional_sample(conditions, returns=returns) 
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)
            head1_out = trainer.ema_model.inv_head1(action)
            head2_out = trainer.ema_model.inv_head2(action)
            gumbel_softmax_output = F.gumbel_softmax(head2_out, tau=1, hard=True)
            action = head1_out * gumbel_softmax_output

            samples = to_np(samples)
            action = to_np(action)

            action = dataset.normalizer.unnormalize(action, 'actions')

            action_low, action_high = process_action_space(env_obs)
            interact_action, _ = process_action(env_obs , action[0], action_low, action_high, env)
            this_obs, this_reward, this_done, info = env.step(interact_action) 
            # logger.print(f"timestep:{timestep},this_reward:{this_reward},this_done:{this_done}")
            origin_obs, _ = state_processer(this_obs)
            episode_reward += this_reward

            if this_done:
                logger.print(f"Episode ({i}): {episode_reward}, rounds: {timestep}", color='green')
                logger.print(f"----------- info: {info['exception']} ---------------")
                rounds.append(timestep)
                episode_rewards.append(episode_reward)
                break
    
    logger.print(f"mean round: {sum(rounds)/len(rounds)}")
    logger.print(f"mean rtgs: {sum(episode_rewards)/len(episode_rewards)}")
        
    # logger.save_pkl() TODO

