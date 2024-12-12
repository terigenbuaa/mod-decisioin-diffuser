import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger
from utils import set_seed, set_config_for_env
from diffuser.utils.arrays import to_torch

from diffuser.datasets.grid import make_env, state_processer, grid_config, my_agent

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        eval_in_train=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        self.eval_in_train = eval_in_train

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        round_best, rtgs_best = 0, 0
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                # evaluate in training
                if self.eval_in_train:
                    round, rtgs = self.eval()
                    if round_best < round:
                        logger.print(f'best round found! step:{self.step}')
                        self.save()
                else:
                    self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')

            if self.step == 0 and self.sample_freq:
                pass
                # self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
                    # self.inv_render_samples()
                    pass
                elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
                    pass
                else:
                    # self.render_samples()
                    pass

            self.step += 1

    def eval(self, rounds_num=20):
        num_eval = 1
        device = self.device
        episode_rewards = []
        rounds = []

        # encode_model checkpoint
        # checkpoint_path = '/home/LAB/terigen/grid2op_mod/grid2op/MakeEnv/checkpoint_epoch_50_encoded_dim_64.pth'

        for i in range(rounds_num):
            episode_reward = 0
            param = set_config_for_env(grid_config)
            env = make_env(param, grid_config)

            # assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
            returns = to_device(0.9 * torch.ones(num_eval, 1), device) # TODO hard coded 0.9 test_ret

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
                origin_obs = self.dataset.normalizer.normalize(origin_obs, 'observations')

                # obs = encode_data(encode_model, to_torch(origin_obs))
                obs = origin_obs
                # import pdb; pdb.set_trace()
                
                conditions = {0: to_torch(obs, device=device)}
                samples = self.ema_model.conditional_sample(conditions, returns=returns) 
                obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2*self.dataset.observation_dim)
                action = self.ema_model.inv_model(obs_comb)

                samples = to_np(samples)
                action = to_np(action)

                action = self.dataset.normalizer.unnormalize(action, 'actions')

                action_low, action_high = my_agent.process_action_space(env_obs)
                interact_action, _ = my_agent.process_action(obs=env_obs , action=action[0], action_low=action_low, action_high=action_high)
                this_obs, this_reward, this_done, info = env.step(interact_action) 
                logger.print(f"timestep:{timestep},this_reward:{this_reward},this_done:{this_done}")
                origin_obs, _ = state_processer(this_obs)
                episode_reward += this_reward

                if this_done:
                    logger.print(f"Episode ({i}): {episode_reward}, rounds: {timestep}", color='green')
                    logger.print(f"----------- info: {info['exception']} ---------------")
                    rounds.append(timestep)
                    episode_rewards.append(episode_reward)
                    break
        
            mean_round = sum(rounds)/len(rounds)
            mean_rtgs = sum(episode_rewards)/len(episode_rewards)
            logger.print(f"mean round: {mean_round}")
            logger.print(f"mean rtgs: {mean_rtgs}")
            return mean_round, mean_rtgs
    
    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')


    def load(self, step=None):
        '''
            loads model and ema from disk
        '''
        if step is None:
            step = self.get_max_step()
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state_{step}.pt')
        # import pdb; pdb.set_trace()
        # data = logger.load_torch(loadpath)
        try:
            data = torch.load(loadpath)
        except FileNotFoundError:
            logger.print(f"Error: Could not find checkpoint file at {loadpath}", color='red')
            raise
        except Exception as e:
            logger.print(f"Error loading checkpoint: {str(e)}", color='red')
            raise
        logger.print(f"model loaded from path: {loadpath}")

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
         # Also load optimizer state if it was saved
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])

        # self.model.encode_model = load_encode_model(64, os.path.join(self.bucket, logger.prefix, f'checkpoint/encode_model_epoch_{step}_dim_64.pth'))

        self.epoch = self.step // 10000

        logger.print(f"load done!!!")

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)