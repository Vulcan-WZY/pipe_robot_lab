import os
import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.utils.spaces.torch import compute_space_size


class NetworkDiagnostics:
    """网络健康监控器: 记录梯度范数、激活值统计、输入数据到 TensorBoard."""
    
    def __init__(self, enabled=False, log_interval=50):
        self.enabled = enabled
        self.log_interval = log_interval
        self._call_count = 0
        self._writer = None
        self._last_vision_input = None
        self._last_vision_features = None
        self._last_prop_input = None
        self._last_prop_features = None
    
    def set_log_dir(self, log_dir):
        if self._writer is None and self.enabled:
            from torch.utils.tensorboard import SummaryWriter as TBWriter
            self._writer = TBWriter(log_dir=log_dir)
    
    def set_writer(self, writer):
        pass
    
    def on_forward(self, img_input, vision_features, prop_input, prop_features, global_step=None):
        if not self.enabled:
            return
        self._call_count += 1
        self._last_vision_input = img_input.detach()
        self._last_vision_features = vision_features.detach()
        self._last_prop_input = prop_input.detach()
        self._last_prop_features = prop_features.detach()
    
    def should_log(self):
        return self.enabled and self._call_count % self.log_interval == 0
    
    def log_to_writer(self, model, global_step):
        if not self.enabled or self._writer is None:
            return
        
        if self._last_vision_features is not None:
            self._writer.add_scalar("Diagnostics/activation_mean_vision", 
                                    self._last_vision_features.mean().item(), global_step)
            self._writer.add_scalar("Diagnostics/activation_std_vision", 
                                    self._last_vision_features.std().item(), global_step)
        
        if self._last_prop_features is not None:
            self._writer.add_scalar("Diagnostics/activation_mean_prop", 
                                    self._last_prop_features.mean().item(), global_step)
            self._writer.add_scalar("Diagnostics/activation_std_prop", 
                                    self._last_prop_features.std().item(), global_step)
        
        if self._last_vision_input is not None:
            self._writer.add_scalar("Diagnostics/input_mean_depth", 
                                    self._last_vision_input.mean().item(), global_step)
            self._writer.add_scalar("Diagnostics/input_nonzero_ratio", 
                                    (self._last_vision_input.abs() > 1e-6).float().mean().item(), global_step)
            img_to_log = self._last_vision_input[0:1].clone()
            for ch in range(img_to_log.shape[1]):
                ch_data = img_to_log[:, ch:ch+1, :, :]
                ch_min = ch_data.min()
                ch_max = ch_data.max()
                if ch_max - ch_min > 1e-6:
                    img_to_log[:, ch:ch+1, :, :] = (ch_data - ch_min) / (ch_max - ch_min)
                else:
                    img_to_log[:, ch:ch+1, :, :] = 0.0
            self._writer.add_images("Diagnostics/depth_input_ch0_front", 
                                    img_to_log[:, 0:1, :, :], global_step)
            self._writer.add_images("Diagnostics/depth_input_ch1_back", 
                                    img_to_log[:, 1:2, :, :], global_step)
        
        cnn_grad_norm = 0.0
        prop_grad_norm = 0.0
        fusion_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if "cnn" in name or "vision_proj" in name:
                    cnn_grad_norm += grad_norm ** 2
                elif "proprioception_mlp" in name:
                    prop_grad_norm += grad_norm ** 2
                elif "fusion_mlp" in name or "output_layer" in name:
                    fusion_grad_norm += grad_norm ** 2
        
        self._writer.add_scalar("Diagnostics/grad_norm_cnn", cnn_grad_norm ** 0.5, global_step)
        self._writer.add_scalar("Diagnostics/grad_norm_prop_mlp", prop_grad_norm ** 0.5, global_step)
        self._writer.add_scalar("Diagnostics/grad_norm_fusion", fusion_grad_norm ** 0.5, global_step)
        self._writer.flush()


_GLOBAL_DIAGNOSTICS = None

def get_diagnostics():
    global _GLOBAL_DIAGNOSTICS
    if _GLOBAL_DIAGNOSTICS is None:
        _GLOBAL_DIAGNOSTICS = NetworkDiagnostics(
            enabled=os.environ.get("PIPE_ROBOT_DEBUG", "0") == "1",
            log_interval=int(os.environ.get("PIPE_ROBOT_DEBUG_INTERVAL", "50"))
        )
    return _GLOBAL_DIAGNOSTICS


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
        self._cam_h = None
        self._cam_w = None

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
            cam_h, cam_w = 60, 80
        
        self._cam_h = cam_h
        self._cam_w = cam_w
            
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
        
        self._prop_dim = policy_dim
        
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

    @staticmethod
    def _sanitize_prop_input(t):
        """检测本体观测中的 NaN/Inf 并替换为 0，返回 (安全张量, 是否含NaN, 是否含Inf)。"""
        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any()
        if has_nan or has_inf:
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t, has_nan, has_inf

    def _maybe_save_action_nan_trace(self, actions, has_nan, has_inf):
        """首次检测到动作 NaN 时保存完整张量及中间特征快照。"""
        if not hasattr(self, "_act_trace_saved"):
            self._act_trace_saved = False
        if self._act_trace_saved:
            return
        self._act_trace_saved = True

        import os as _os
        import time as _time
        trace_dir = _os.environ.get("PIPE_ROBOT_NAN_TRACE_DIR", "")
        if not trace_dir:
            return

        ts = _time.strftime("%Y%m%d_%H%M%S")
        fpath = _os.path.join(trace_dir, f"act_nan_{ts}.pt")
        pipe_txt_path = _os.path.join(trace_dir, f"{ts}.txt")
        try:
            snapshot = {
                "actions": actions.detach().cpu(),
                "action_shape": tuple(actions.shape),
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "nan_count": int(torch.isnan(actions).sum().item()),
                "nan_dims": torch.isnan(actions).nonzero(as_tuple=False)[:30].cpu().tolist(),
                "log_std": self.log_std_parameter.detach().cpu(),
            }
            import logging
            _logger = logging.getLogger(__name__)
            _logger.info(f"[NAN-TRACE] Saved action snapshot to {fpath} "
                         f"(NaN count: {snapshot['nan_count']})")
            torch.save(snapshot, fpath)
            selected_pipe = _os.environ.get("PIPE_ROBOT_SELECTED_PIPE_USD", "")
            with open(pipe_txt_path, "w", encoding="utf-8") as f:
                f.write(selected_pipe)
        except Exception:
            pass

    def compute(self, inputs, role=""):
        obs_dict = inputs.get("observations", inputs.get("states", None))
        if torch.is_tensor(obs_dict):
            from skrl.utils.spaces.torch import unflatten_tensorized_space
            obs_dict = unflatten_tensorized_space(self.observation_space, obs_dict)

        batch_size = next(iter(inputs.values())).shape[0] if torch.is_tensor(next(iter(inputs.values()))) else 1

        camera_data = obs_dict.get("camera", None) if isinstance(obs_dict, dict) else None
        if camera_data is not None and isinstance(camera_data, dict):
            img_front = camera_data.get("depth_front", None)
            img_back = camera_data.get("depth_back", None)
        else:
            img_front = None
            img_back = None

        if img_front is None or img_back is None:
            device = self.output_layer.weight.device
            img_front = torch.zeros(batch_size, 1, self._cam_h, self._cam_w, device=device)
            img_back = torch.zeros(batch_size, 1, self._cam_h, self._cam_w, device=device)
        else:
            img_front = torch.nan_to_num(img_front, nan=0.0, posinf=10.0, neginf=0.0)
            img_back = torch.nan_to_num(img_back, nan=0.0, posinf=10.0, neginf=0.0)
            img_front = torch.clamp(img_front, 0.0, 10.0) / 10.0
            img_back = torch.clamp(img_back, 0.0, 10.0) / 10.0

        img_input = torch.cat([img_front, img_back], dim=1)
        vision_features = self.vision_proj(self.cnn(img_input))

        if self.is_critic and isinstance(obs_dict, dict) and "critic" in obs_dict:
            prop_input = obs_dict["critic"]
        elif not self.is_critic and isinstance(obs_dict, dict) and "policy" in obs_dict:
            prop_input = obs_dict["policy"]
        else:
            device = self.output_layer.weight.device
            prop_input = torch.zeros(batch_size, self._prop_dim, device=device)

        # NaN/Inf 检测：本体观测被污染时（如仿真矩阵退化）替换为安全值
        prop_input, prop_has_nan, prop_has_inf = self._sanitize_prop_input(prop_input)

        prop_features = self.proprioception_mlp(prop_input)

        get_diagnostics().on_forward(img_input, vision_features, prop_input, prop_features)

        fused_features = torch.cat([vision_features, prop_features], dim=-1)
        latent_features = self.fusion_mlp(fused_features)

        if not self.is_critic:
            actions = self.output_layer(latent_features)
            # 动作 NaN/Inf 检测：输出异常时归零，防止污染仿真加剧退化
            act_has_nan = torch.isnan(actions).any()
            act_has_inf = torch.isinf(actions).any()
            if act_has_nan or act_has_inf:
                if not hasattr(self, "_action_nan_count"):
                    self._action_nan_count = 0
                    import logging
                    self._action_logger = logging.getLogger(__name__)
                if self._action_nan_count < 10:
                    nan_idxs = torch.isnan(actions).nonzero(as_tuple=False)
                    idx_str = ", ".join(str(idx.tolist()) for idx in nan_idxs[:20])
                    if nan_idxs.shape[0] > 20:
                        idx_str += f", ... ({nan_idxs.shape[0]} total)"
                    self._action_logger.warning(
                        f"[ACT-NaN] NaN/Inf detected in Actor output (dims: [{idx_str}]), "
                        f"replaced with 0. (warning {self._action_nan_count + 1}/10)"
                    )
                    self._action_nan_count += 1
                # NaN trace 快照保存
                self._maybe_save_action_nan_trace(actions, act_has_nan, act_has_inf)
                actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            return actions, {"log_std": self.log_std_parameter}
        else:
            value = self.output_layer(latent_features)
            if torch.isnan(value).any() or torch.isinf(value).any():
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value, {}

class CustomActor(GaussianMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space=observation_space, action_space=action_space, device=device, is_critic=False, **kwargs)
        GaussianMixin.__init__(self, clip_actions=False)

class CustomCritic(DeterministicMixin, CustomActorCritic):
    def __init__(self, observation_space, action_space, device, **kwargs):
        CustomActorCritic.__init__(self, observation_space=observation_space, action_space=action_space, device=device, is_critic=True, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)
