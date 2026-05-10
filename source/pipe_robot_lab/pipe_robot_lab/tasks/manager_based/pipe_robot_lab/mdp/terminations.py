# ===========
# Date: 2026-01-30 11:02
# Author: Vulcan
# LastEditTime: 2026-05-06 13:09
# Description: 配置训练环境的终止检测函数
# ==========
import torch
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

import pipe_robot_lab.funcs.pipe_geom_utils as p_geom

def auto_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    distance_threshold: float = 0.05,
    max_steps: int = 100,
    min_steps: int = 0,
    enabled: bool = True,
) -> torch.Tensor:
    """
    停滞自检函数, 检测到前后轮刚体在一系列步数内的位移,如果两者在这段时间内没有移出指定的范围
    则认为机器人卡死, 触发重置信号
    """
    if not enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset = env.scene[asset_cfg.name] # 拿机器人资产
    body_ids = asset.find_bodies(asset_cfg.body_names)[0] # 获取刚体的索引
    current_pos = asset.data.body_pos_w[:,body_ids,:] # 避免 clone() 原地分配额外显卡内存
    
    if not hasattr(env, "_stagnation_pos_history"):
        env._stagnation_pos_history = current_pos.clone() # 首次初始化脱离引用
        env._stagnation_step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # 获取坐标变化距离, 使用欧氏距离
    diff = current_pos - env._stagnation_pos_history
    dist = torch.norm(diff, p = 2, dim = -1) # shape : [num_envs , 2]
    
    # 判断是否在有效移动中
    moved = torch.any(dist > distance_threshold, dim = -1) # shape: [num_envs]
    
    # 状态更新逻辑:
    # 发生移动的env, 清零计时器; 反之计数器+1
    env._stagnation_step_count = torch.where(
        moved,
        torch.zeros_like(env._stagnation_step_count),
        env._stagnation_step_count + 1
    )    
    # 发生移动的env, 将当前的基准点重置为当前位置
    moved_expanded = moved.unsqueeze(-1).unsqueeze(-1).expand_as(current_pos)
    env._stagnation_pos_history = torch.where(moved_expanded, current_pos, env._stagnation_pos_history)

    # 回合早期不触发卡死，并在早期阶段重置计数器/基准点，避免跨回合遗留导致“step=1秒重置”。
    if min_steps > 0:
        too_young = env.episode_length_buf < min_steps
        env._stagnation_step_count = torch.where(
            too_young,
            torch.zeros_like(env._stagnation_step_count),
            env._stagnation_step_count,
        )
    too_young_expanded = too_young.unsqueeze(-1).unsqueeze(-1).expand_as(current_pos)
    env._stagnation_pos_history = torch.where(too_young_expanded, current_pos, env._stagnation_pos_history)
    
    # 当连续没按要求移出范围的步数大于设定值时,触发终止
    stagnated = env._stagnation_step_count >= max_steps
    if min_steps > 0:
        stagnated = torch.where(too_young, torch.zeros_like(stagnated), stagnated)
    return stagnated

def no_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_names: list,
    threshold: float = 0.01,
    min_steps: int = 15,
    enabled: bool = True,
) -> torch.Tensor:
    """
    自定义终止函数: 检测是否所有的接触力传感器都没有接触到管道. 
    增加 min_steps 参数：在环境重置后的前 min_steps 步内，即使完全脱离也不判定为终止。
    返回布尔张量[num_envs], 当所有传感器的接触力都小于或等于threshold时为True
    """
    # 通过开关临时失能该终止项，同时保留 term 名称供其他模块引用。
    if not enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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

def reach_goal_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_steps: int = 0,
    enabled: bool = True,
) -> torch.Tensor:
    """
    终点判断终止函数: 提取机器人的特定部位(如前侧基座)在世界下的坐标，
    判断其是否进入了最后一段管道并且行进距离超过了该段的一半。
    """
    if not enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    num_segments = len(env.cfg.pipe_transform_inv)  # 一共有几段管元
    # 调用现有的缓存方法：返回 [num_envs, 5] -> (x_local, y_local, z_local, seg_id + ratio, radial_ratio)
    out = p_geom.get_cached_pipe_info(env, asset_cfg)
    # 提取进度。有效值中 progress 的整数部分代表走到的管段 index(0-based), 小数部分是占比
    progress = out[:, 3]
    # 触发重置的阈值：到达最后一段 (num_segments - 1) 的给定比例以上, 即 num_segments - 0.5
    threshold = (num_segments - 1) + 0.8
    # -1 代表完全找不到对应管断，必须排除
    reach_goal = torch.logical_and(progress >= threshold, progress != -1.0)
    if min_steps > 0:
        too_young = env.episode_length_buf < min_steps
        reach_goal = torch.where(too_young, torch.zeros_like(reach_goal), reach_goal)
    return reach_goal

@configclass
class TerminationsCfg:
    # 终止函数需要返回一个布尔张量`num_envs`
    
    # 1. 基础超时终止 (配合 episode_length_s)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. 6个轮子全部脱离接触即重置
    
    lost_all_contacts = DoneTerm(
        func=no_contact_termination,
        params={
            "sensor_names": ["touch_m1", "touch_m2", "touch_a1", "touch_a2", "touch_a3", "touch_a4"],
            "threshold": 0.05,
            "min_steps": 50,
            "enabled": True,
        }
    )
    
    # 3. 成功跑到终点的一半以上（完成任务重启）
    reach_goal_reset = DoneTerm(
        func=reach_goal_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"]),
            "min_steps": 100,
            "enabled": False,
        }
    )
    
    # 4. 卡死停滞判断终止 (动态管网专用的终止)
    stagnation_reset = DoneTerm(
        func=auto_termination,
        params={
            # 填入你刚才提供的前后轮坐标
            "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link", "BM_01_link"]),
            "distance_threshold": 0.08,    # 最大漂移范围：0.05米（即5cm）
            "max_steps": 120,    # 卡死时限：依据你的dt（例如100个step = 1秒，这里设成两秒半左右）
            "min_steps": 30,
            "enabled": False,
        }
    )