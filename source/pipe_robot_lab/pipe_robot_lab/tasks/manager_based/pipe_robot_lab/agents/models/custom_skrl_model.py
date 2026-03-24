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

class CustomActorCritic(Model):
    def __init__(self, observation_space, action_space, device, is_critic=False):
        # 1. 初始化 Model 基类
        Model.__init__(self, observation_space, action_space, device)
        
        # 2. 如果包含多个角色，比如同时被用作 Actor 和 Critic (如果是单独的，可以在 kwargs 里传 role)
        # 这里为了演示和 SKRL 集成更加简单，我们可以根据 output_shape 或者 kwargs 里的标识判断
        # 一般在 skrl 中，可以用单独的两个类（Actor和Critic），或者用一个共享的骨干网络。
        # 此处使用两套分别初始化的思路（通过参数区分）：
        self.is_critic = is_critic

        # ====== A. 视觉特征提取器 (CNN) ======
        # 深度图：前后各一个，我们假设拼接在一起后是 2 通道
        # 常见图像大小假定为 64x64 或者稍微更大一点（需根据你 TiledCamera 的配置而定）
        # 这里构建一个轻量级的 CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        # 自动推导 CNN 平铺维度，避免写死分辨率导致线性层尺寸错误
        camera_space = _get_subspace(observation_space, "camera")
        front_depth_space = _get_subspace(camera_space, "depth_front") if camera_space is not None else None
        if front_depth_space is not None and hasattr(front_depth_space, "shape") and len(front_depth_space.shape) == 3:
            cam_h = front_depth_space.shape[1]
            cam_w = front_depth_space.shape[2]
        else:
            cam_h, cam_w = 64, 64
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, 2, cam_h, cam_w)).shape[-1]

        # ====== B. 本体感受特征提取器 (MLP) ======
        # 本体维度来源于你的 PolicyCfg 展平后的总维度
        # 在这里我们暂时设一个泛泛的值，可以通过 observation_space 来动态获取维度
        # SKRL 在字典空间时， observation_space 对应字典
        policy_space = _get_subspace(observation_space, "policy")
        if policy_space is not None:
            policy_dim = compute_space_size(policy_space)
        else:
            policy_dim = 100
        
        # Critic 直接使用 critic 观测组，不再和 policy 重复拼接
        if self.is_critic:
            critic_space = _get_subspace(observation_space, "critic")
            if critic_space is not None:
                policy_dim = compute_space_size(critic_space)
        
        self.proprioception_mlp = nn.Sequential(
            nn.Linear(policy_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        # ====== C. 融合与输出层 ======
        fusion_dim = cnn_out_dim + 128
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        
        # ====== D. 动作输出 / 价值输出 ======
        if not self.is_critic:
            # Actor: 输出均值 (Mean)，动作空间大小对应 action_space
            action_dim = compute_space_size(action_space)
            self.output_layer = nn.Linear(128, action_dim)
            # 在 GaussianMixin 中，标准差 (std) 通常作为可学习的参数 (Parameter) 处理
            self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        else:
            # Critic: 输出状态价值 (Value) -> 1 维
            self.output_layer = nn.Linear(128, 1)

    def compute(self, inputs, role=""):
        """
        SKRL Model 要求实现 compute 方法
        inputs["states"] 是你的包含多模态数据的字典
        """
        obs_dict = inputs["states"]
        if torch.is_tensor(obs_dict):
            obs_dict = self.tensor_to_space(obs_dict, self.observation_space)

        # 1. 取出图像并拼接
        img_front = obs_dict["camera"]["depth_front"]
        img_back = obs_dict["camera"]["depth_back"]
        # 在通道维度拼接：要求形状为 [Num_Envs, Channel, Height, Width]
        # 此时应该变为 [N, 2, H, W]
        img_input = torch.cat([img_front, img_back], dim=1)
        
        # 前向 CNN
        vision_features = self.cnn(img_input)

        # 2. 取出本体感受
        if self.is_critic and "critic" in obs_dict:
            prop_input = obs_dict["critic"]
        else:
            prop_input = obs_dict["policy"]
        
        # 前向 MLP
        prop_features = self.proprioception_mlp(prop_input)

        # 3. 特征融合
        fused_features = torch.cat([vision_features, prop_features], dim=-1)
        latent_features = self.fusion_mlp(fused_features)

        # 4. 输出
        if not self.is_critic:
            # 返回 (Mean, log_std, {} ) 给 GaussianMixin
            # 对于 GaussianMixin，如果定义了 self.log_std_parameter 并返回 log_std
            # skrl会自动处理正态分布
            return self.output_layer(latent_features), self.log_std_parameter, {}
        else:
            # Critic 输出 V(s)
            return self.output_layer(latent_features), {}

class CustomActor(GaussianMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space, action_space, device, is_critic=False)
        GaussianMixin.__init__(self, clip_actions=False)

class CustomCritic(DeterministicMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space, action_space, device, is_critic=True)
        DeterministicMixin.__init__(self, clip_actions=False)
