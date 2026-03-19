# ===========
# Date: 2026-01-30 11:02
# Author: Vulcan
# LastEditTime: 2026-03-17 15:12
# Description: 配置训练环境的终止检测函数
# ==========
import torch
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

import pipe_robot_lab.funcs.pipe_geom_utils as p_geom

def no_contact_termination(env: ManagerBasedRLEnv, sensor_names: list , threshold: float = 0.01, min_steps: int = 15) -> torch.Tensor:
    """
    自定义终止函数: 检测是否所有的接触力传感器都没有接触到管道. 
    增加 min_steps 参数：在环境重置后的前 min_steps 步内，即使完全脱离也不判定为终止。
    返回布尔张量[num_envs], 当所有传感器的接触力都小于或等于threshold时为True
    """
    # 初始假设所有环境都没有接触 (True)
    all_detached = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in sensor_names:
        # 获取具体的传感器对象
        sensor = env.scene.sensors[sensor_name]
        # sensor.data.net_forces_w 的形状一般为 [num_envs, 3] 或者 [num_envs, body_num, 3]
        # 计算合力的大小
        force_mag = torch.norm(sensor.data.net_forces_w, dim=-1, p=2) 
        # force_mag 可能含有多余维度如 [num_envs, 1], 取 .view(-1) 压平为 [num_envs]
        force_mag = force_mag.view(env.num_envs)
        # 只要有一点受力大于阈值，说明该轮子还在接触
        is_contacting = force_mag > threshold
        # 逻辑与运算：只有当之前的轮子全都脱离，且当前这个轮子也脱离时，all_detached 才会继续保持为 True
        all_detached = torch.logical_and(all_detached, ~is_contacting)
        
    # 【关键改进】：屏蔽掉在前 `min_steps` 步的环境，防止刚初始化在空中就触发重置
    too_young = env.episode_length_buf < min_steps
    # 对于步数不够的环境，强制将它们的终止信号掩码置为 False
    all_detached = torch.where(too_young, torch.zeros_like(all_detached), all_detached)
        
    return all_detached

def reach_goal_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    终点判断终止函数: 提取机器人的特定部位(如前侧基座)在世界下的坐标，
    判断其是否进入了最后一段管道并且行进距离超过了该段的一半。
    """
    num_segments = env.cfg.pipe_transform_inv.shape[0]  # 一共有几段管元
    
    # 调用现有的缓存方法：返回 [num_envs, 5] -> (x_local, y_local, z_local, seg_id + ratio, radial_ratio)
    out = p_geom.get_cached_pipe_info(env, asset_cfg)
    
    # 提取进度。有效值中 progress 的整数部分代表走到的管段 index(0-based), 小数部分是占比
    progress = out[:, 3]
    # 触发重置的阈值：到达最后一段 (num_segments - 1) 的给定比例以上, 即 num_segments - 0.5
    threshold = (num_segments - 1) + 0.8
    # -1 代表完全找不到对应管断，必须排除
    reach_goal = torch.logical_and(progress >= threshold, progress != -1.0)
    return reach_goal

@configclass
class TerminationsCfg:
    # 终止函数需要返回一个布尔张量`num_envs`
    # 1. 基础超时终止 (回合最大步数到了自动终止，必须保留)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 2. 我们刚才设计的：6个轮子全部脱离接触即重置
    lost_all_contacts = DoneTerm(
        func=no_contact_termination,
        params={
            # 传入你在 scene 中注册的那6个接触传感器的名称
            "sensor_names": ["touch_m1", "touch_m2", "touch_a1", "touch_a2", "touch_a3", "touch_a4"],
            "threshold": 0.05,  # 设置一个小的力阈值（牛顿），避免受到数值误差干扰
            "min_steps": 100     # 给机器人 20 步的自由落体和与管道贴合的时间
        }
    )
    # 3. 成功跑到终点的一半以上（完成任务重启）
    reach_goal_reset = DoneTerm(
        func=reach_goal_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])
        }
    )