import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb
import os
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from ml_logger import logger

from diffuser.datasets.data_encoder_decoder import load_encode_model, encode_data, decode_data
from diffuser.datasets.grid import make_env, state_processer, grid_config, my_agent

grid_config.balance_mid_val = torch.tensor(grid_config.mid_val, dtype=torch.float32).to(grid_config.device)


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False, load_pre_encoder=True, encoded_dim=16):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)
        # import pdb; pdb.set_trace()

        if load_pre_encoder:
            # Example usage
            encoded_dim = observation_dim # Use the best encoded_dim found during training
            home_dir = os.path.expanduser("~")
            checkpoint_path = os.path.join(home_dir, "/home/LAB/terigen/PyTorch-VAE/logs-new-grid/GridVAE/GridVAE_lr0.001_lat64_kld0.01/checkpoints/last.ckpt")  
            # checkpoint_path = os.path.join(home_dir, "grid2op_mod/grid2op/MakeEnv/checkpoint_epoch_50_encoded_dim_64.pth")
            self.encode_model = load_encode_model(encoded_dim, checkpoint_path)  
            logger.print(f"load encode model from {checkpoint_path}")  

            # freeze encode model
            # for param in self.encode_model.parameters():
            #     param.requires_grad = False
                   
            self.observation_dim = encoded_dim
            self.current_epoch = None

    # def update_encoder_freeze(self, epoch):
    #     self.current_epoch = epoch
    #     if epoch < 4:
    #         # Keep all layers frozen for the first 7 epochs
    #         return
    #     elif epoch < 5:
    #         for param in self.encode_model.embedding.parameters():
    #             param.requires_grad = True
    #             logger.print(f"Unfroze: Updated Embedding Layer: {param.requires_grad}")
    #         return

    #     i = 4
    #     for name, param in reversed(list(self.encode_model.transformer.encoder.named_parameters())):
    #         i += 1
    #         if i - epoch <  0:
    #             param.requires_grad = True 
    #             logger.print(f"Unfroze: i: {i}; Updated Layer {name}: {param.requires_grad}") 
    #         else:              
    #             break

    def update_encoder_freeze(self, epoch):
        self.current_epoch = epoch
        if epoch < 4:
            # Keep all layers frozen for the first 3 epochs
            return
        
        # First unfreeze encoder layers gradually (epochs 4-7)
        i = 4
        for name, param in reversed(list(self.encode_model.transformer.encoder.named_parameters())):
            i += 1
            if i - epoch < 0:
                param.requires_grad = True 
                logger.print(f"Unfroze: i: {i}; Updated Encoder Layer {name}: {param.requires_grad}") 
            else:              
                break
        
        # Then unfreeze embedding layer at epoch 8
        if epoch >= 55:
            for name, param in self.encode_model.embedding.named_parameters():
                param.requires_grad = True
                logger.print(f"Unfroze: Updated Embedding Layer {name}: {param.requires_grad}")
        
        # Finally unfreeze positional encoder at epoch 9
        if epoch >= 57:
            for param in self.encode_model.positional_encoding:
                param.requires_grad = True
                logger.print(f"Unfroze: Updated Positional Encoder Layer: {param.requires_grad}")

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        cond = encode_data(self.encode_model, cond) # shape of cond[0]: [32, 409]
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # shape: [32, 100, 64]
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        obs = x_noisy - x_recon
        # decode obs to 409 dim
        # print("Before decode_data:")
        # print(f"obs shape: {obs.shape}")
        # print(f"obs requires_grad: {obs.requires_grad}")
        # print(f"obs version: {obs._version}")
        
        decoded_obs = decode_data(self.encode_model, obs)
        
        # print("After decode_data:")
        # print(f"decoded_obs shape: {decoded_obs.shape}")
        # print(f"decoded_obs requires_grad: {decoded_obs.requires_grad}")
        # print(f"decoded_obs version: {decoded_obs._version}")
        
        # assert decoded_obs.shape == (32, 100, 409)
        x_t = obs[:, :-1, :]
        x_t_1 = obs[:, 1:, :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        pred_a = self.inv_model(x_comb_t)

        # get pred_a by inv_model from generated x_t and x_t_1
        # balance_loss = self.calculate_balance_loss(decoded_obs, pred_a.reshape(grid_config.batch_size, 99, self.action_dim))

        return loss, info #, balance_loss

    def loss(self, epoch, x, cond, returns=None):
        if self.current_epoch != epoch: 
            new_epoch = True
            self.current_epoch = epoch
        else:
            new_epoch = False

        if new_epoch:
            # self.update_encoder_freeze(self.current_epoch)
            pass

        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            obs = x[:,:,self.action_dim:]
            # encode observation
            x_encoded = encode_data(self.encode_model, obs)
            # x[:,:,self.action_dim:] = x_encoded
            # diffuse_loss, info, balance_loss = self.p_losses(x_encoded, cond, t, returns)
            diffuse_loss, info = self.p_losses(x_encoded, cond, t, returns)
            # Calculating inv loss
            x_t = x_encoded[:, :-1, :]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x_encoded[:, 1:, :]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t) 
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            # balance_loss = self.calculate_balance_loss(obs, pred_a_t.reshape(batch_size, 99, self.action_dim))

            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info
    
    def calculate_balance_loss(self, obs_horizon, action_horizon):
        # 目前只取了整个horizon的第一个元素
        obs = obs_horizon[:, 0, :]
        action = action_horizon[:, 0, :].clone()
        # set balance redispatch to 0
        action[:, :grid_config.n_gen][:, grid_config.name_gen == 'balance'] = 0.
        # split action by fixed position
        adjust_gen_p = action[:, grid_config.detail_action_dim[0][0]: grid_config.detail_action_dim[0][1]]
        adjust_storage_p = action[:, grid_config.detail_action_dim[1][0]: grid_config.detail_action_dim[1][1]]
        # split obs by fixed position
        balance_p, gen_p, load_p, storage_p = self.split_obs(obs, move_balance=True)
        # print('adjust_gen_p', adjust_gen_p.shape)
        # print('adjust_storage_p', adjust_storage_p.shape)
        # print('gen_p', gen_p.shape)
        # print('load_p', load_p.shape)
        # print('storage_p', storage_p.shape)
        # print('balance_p', balance_p.shape)
        predict_gen_p = torch.sum(gen_p, dim=-1) + torch.sum(adjust_gen_p, dim=-1)
        predict_load_p = torch.sum(load_p, dim=-1)
        predict_storage_p = torch.sum(adjust_storage_p, dim=-1)
        predict_loss = balance_p.squeeze(-1) + torch.sum(gen_p, dim=-1) - torch.sum(load_p, dim=-1) - torch.sum(adjust_storage_p, dim=-1)
        predict_balance_p = predict_load_p + predict_storage_p + predict_loss - predict_gen_p
        # print('-------------------------------')
        # print('predict_gen_p', predict_gen_p)
        # print('predict_load_p', predict_load_p)
        # print('predict_storage_p', predict_storage_p)
        # print('predict_loss', predict_loss)
        # print('predict_balance_p', predict_balance_p)
        # print('balance_p', balance_p.squeeze(-1))
        # print('balance_p', balance_p.squeeze(-1) - torch.sum(adjust_gen_p, dim=-1))
        if grid_config.split_balance_loss:
            balance_loss_rate = torch.where((predict_balance_p >= grid_config.warning_region_lower) &
                                            (predict_balance_p <= grid_config.warning_region_upper),
                                            grid_config.warning_balance_loss_rate, grid_config.danger_balance_loss_rate)
            balance_loss_rate = torch.where((predict_balance_p >= grid_config.save_region_lower) &
                                            (predict_balance_p <= grid_config.save_region_upper),
                                            grid_config.save_balance_loss_rate, balance_loss_rate)
        else:
            balance_loss_rate = grid_config.balance_loss_rate
        # not_dones = (-dones + 1).squeeze(-1)
        # return (((next_predict_balance_p - self.balance_mid_val) ** 2) * balance_loss_rate * not_dones).mean()
        # print('predict_balance_p', predict_balance_p)
        # print('balance_loss_rate', balance_loss_rate)
        return (((predict_balance_p - grid_config.balance_mid_val) ** 2) * balance_loss_rate).mean()
    
    def split_obs(self, obs, move_balance=True):
        gen_p = obs[:, 0:grid_config.n_gen]
        load_p = obs[:, grid_config.n_gen:
                        grid_config.n_gen + grid_config.n_load]
        storage_power = obs[:, grid_config.n_gen + grid_config.n_load:
                               grid_config.n_gen + grid_config.n_load + grid_config.n_storage] \
            # if grid_config.storage_available else None
        balance_p = gen_p[:, grid_config.name_gen == 'balance']
        gen_p[:, grid_config.name_gen == 'balance'] *= (1. - move_balance)

        return balance_p, gen_p, load_p, storage_power

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

