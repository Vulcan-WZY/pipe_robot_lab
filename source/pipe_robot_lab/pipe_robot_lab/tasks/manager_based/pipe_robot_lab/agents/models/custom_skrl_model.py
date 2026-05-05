import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.utils.spaces.torch import compute_space_size


def _get_subspace(space, key):
    if isinstance(space, dict):
        return space.get(key, None)
    if hasattr(space, "spaces"):
        return space.spaces.get(key, None)
    return None

class ResBlock(nn.Module):
    """用于强化学习的轻量级残差块 (参考 Impala CNN)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.ELU()
        
    def forward(self, x):
        res = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + res)

class CustomActorCritic(Model):
    def __init__(self, observation_space, action_space, device, is_critic=False, **kwargs):
        kwargs.pop("return_source", None)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
        
        self.is_critic = is_critic

        # ====== A. 高级视觉特征提取器 (ResNet-like) ======
        # 相比原先的直连CNN，增加 GroupNorm 和残差连接，大幅提升深度图几何特征的提取能力
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            ResBlock(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),
            ResBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ELU(),
            ResBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ELU(),
            ResBlock(256),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        camera_space = _get_subspace(observation_space, "camera")
        front_depth_space = _get_subspace(camera_space, "depth_front") if camera_space is not None else None
        if front_depth_space is not None and hasattr(front_depth_space, "shape") and len(front_depth_space.shape) == 3:
            cam_h = front_depth_space.shape[1]
            cam_w = front_depth_space.shape[2]
        else:
            cam_h, cam_w = 64, 64
            
        with torch.no_grad():
            cnn_backbone_dim = self.cnn(torch.zeros(1, 2, cam_h, cam_w)).shape[-1]

        # 视觉投影：将 CNN 输出归一化并映射到稳定的潜在子空间
        self.vision_proj = nn.Sequential(
            nn.Linear(cnn_backbone_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )
        cnn_out_dim = 256

        # ====== B. 强表征本体特征提取器 (MLP with LayerNorm) ======
        policy_space = _get_subspace(observation_space, "policy")
        if policy_space is not None:
            policy_dim = compute_space_size(policy_space)
        else:
            policy_dim = 100
        
        if self.is_critic:
            critic_space = _get_subspace(observation_space, "critic")
            if critic_space is not None:
                policy_dim = compute_space_size(critic_space)
        
        # 增加网络深度和宽度，最关键的是加入 LayerNorm 防止不同量纲的状态引爆梯度
        self.proprioception_mlp = nn.Sequential(
            nn.Linear(policy_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU()
        )

        # ====== C. 深度特征融合模块 ======
        fusion_dim = cnn_out_dim + 256 # 256 (Vision) + 256 (Prop) = 512
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU()
        )
        
        # ====== D. 动作输出 / 价值输出 ======
        if not self.is_critic:
            action_dim = compute_space_size(action_space)
            self.output_layer = nn.Linear(256, action_dim)
            self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        else:
            self.output_layer = nn.Linear(256, 1)
            
        self._init_weights()

    def _init_weights(self):
        """PPO正交初始化：让网络早期输出方差大但均值向零靠拢，加速收敛"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 对于 Actor 最后层，将初始输出压缩到接近0，让初期动作由环境纯噪声主导探索
        if not self.is_critic:
            nn.init.orthogonal_(self.output_layer.weight, gain=0.01)

    def compute(self, inputs, role=""):
        obs_dict = inputs["states"]
        if torch.is_tensor(obs_dict):
            # tensor_to_space was incorrectly returning something or nothing.
            from skrl.utils.spaces.torch import unflatten_tensorized_space
            obs_dict = unflatten_tensorized_space(self.observation_space, obs_dict)

        img_front = obs_dict["camera"]["depth_front"]
        img_back = obs_dict["camera"]["depth_back"]

        img_front = torch.nan_to_num(img_front, nan=0.0, posinf=10.0, neginf=0.0)
        img_back = torch.nan_to_num(img_back, nan=0.0, posinf=10.0, neginf=0.0)
        img_front = torch.clamp(img_front, 0.0, 10.0) / 10.0
        img_back = torch.clamp(img_back, 0.0, 10.0) / 10.0
        
        img_input = torch.cat([img_front, img_back], dim=1)
        vision_features = self.vision_proj(self.cnn(img_input))

        if self.is_critic and "critic" in obs_dict:
            prop_input = obs_dict["critic"]
        else:
            prop_input = obs_dict["policy"]
        
        prop_features = self.proprioception_mlp(prop_input)

        fused_features = torch.cat([vision_features, prop_features], dim=-1)
        latent_features = self.fusion_mlp(fused_features)

        if not self.is_critic:
            return self.output_layer(latent_features), self.log_std_parameter, {}
        else:
            return self.output_layer(latent_features), {}

class CustomActor(GaussianMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space=observation_space, action_space=action_space, device=device, is_critic=False, **kwargs)
        GaussianMixin.__init__(self, clip_actions=False)

class CustomCritic(DeterministicMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space=observation_space, action_space=action_space, device=device, is_critic=True, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)
