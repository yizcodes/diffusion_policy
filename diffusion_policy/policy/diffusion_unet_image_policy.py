from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # 调用了 DDPMs Scheduler, 参数在 config 里，policy.noise_scheduler 下
            noise_scheduler: DDPMScheduler, 
            # 调用了 multi_image_obs_encoder.py， 参数在 config 里，diffusion_policy/model/vision/multi_image_obs_encoder.py 初始化方法
            # 获得的 obs_encoder 是多个模型，多个 resnet 的 image encoder
            # 每个 image encoder 有 nn.Sequential(this_resizer, this_randomizer, this_normalizer) + resnet 的操作
            obs_encoder: MultiImageObsEncoder, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes， shape_meta 从 config 传入的，主要是 observation 和 action
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim, 调用了 obs_encoder, 给一个虚拟的输入，获取输出的形状
        # 例如 两个 resnet + action shape = 2 * 512 + 2 = 1026
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model，输入 dimension 是 action dimention + observation dimension (image + low_dim)
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond: # 输入作为一个全局的 condition
            input_dim = action_dim
            # n_obs_steps = 2, 需要两步的 observation
            # 根据论文，这里定义的 horizon 就是论文里的 prediction horizon
            global_cond_dim = obs_feature_dim * n_obs_steps
            
        # input_dim: 输入数据的维度。在动作预测模型中，这可能是动作向量的维度。
        # local_cond_dim 和 global_cond_dim: 分别代表局部条件和全局条件的维度。局部条件可以是与每个时间步骤相关的信息，而全局条件是整个序列共享的信息。
        # diffusion_step_embed_dim: 扩散步骤编码的维度，用于将扩散时间步骤编码为一个连续的特征表示。
        # down_dims, kernel_size, n_groups: 分别定义了U-Net下采样路径中不同层的维度、卷积核大小和分组卷积的组数。
        # cond_predict_scale: 一个布尔值，指示模型是否预测每个通道的缩放因子和偏置项来调制特征图，这是FiLM（Feature-wise Linear Modulation）技术的一种应用。

        # 定义了一个 model
        model = ConditionalUnet1D(
            input_dim=input_dim, # action dim, 2 
            local_cond_dim=None,
            global_cond_dim=global_cond_dim, # bs_feature_dim * n_obs_steps
            diffusion_step_embed_dim=diffusion_step_embed_dim, # diffusion step 经过 encode 后的维度
            down_dims=down_dims, # Unet 下采样路径中不同层的维度
            kernel_size=kernel_size,
            n_groups=n_groups, # n groups 8, group norm 的 group 数
            cond_predict_scale=cond_predict_scale # what is this for?
        )

        self.obs_encoder = obs_encoder # observation encoder
        self.model = model # 预测 action 的 model， 
        self.noise_scheduler = noise_scheduler # diffuser DDPMs, 噪音生成
        # 没有搞懂这个 maskgenerator 是干嘛的 ?
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon # 16
        self.obs_feature_dim = obs_feature_dim # 1026
        self.action_dim = action_dim # 2
        self.n_action_steps = n_action_steps # 8
        self.n_obs_steps = n_obs_steps # 2 
        self.obs_as_global_cond = obs_as_global_cond # true
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps # 100 diffusion step 的步数
    
    # ========= inference  ============
    # 训练过程不需要
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    # 训练过程不需要
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs']) # [64, 2, 3, 240, 320]
        nactions = self.normalizer['action'].normalize(batch['action']) # [64, 16, 2]
        batch_size = nactions.shape[0] # 64
        horizon = nactions.shape[1] # 16

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T 例如 [64, 2, 3, 240, 320] -> [64*2, 3, 240, 320], 方便处理吧
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs) # [128, 1026]
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # [64, 2052]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        # 干嘛用的呢，调用了前面初始化的 mask generator
        # 生成了一个 全是 false 的 mask, [64, 16, 2]
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        # pred [batch, 16, 2], 
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
