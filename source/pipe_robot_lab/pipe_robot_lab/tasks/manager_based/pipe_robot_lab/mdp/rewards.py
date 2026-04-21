# ===========
# Date: 2026-01-11 15:20
# Author: Vulcan
# LastEditTime: 2026-04-21 11:17
# Description: 主要配置管道检测机器人运动时的reward
# ==========
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import pipe_robot_lab.funcs.pipe_geom_utils as p_geom

def axial_progress_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    基于势能的轴向里程奖励：仅在机器人突破历史最大里程时给予奖励
    """
    # 直接调用缓存的数据避免重复计算
    relative_info = p_geom.get_cached_pipe_info(env, asset_cfg)
    # 提取总里程进度 (seg_id + axial_ratio)
    current_progress = relative_info[:, 3] 
    # 1. 动态初始化环境最大里程记录 buffer
    if not hasattr(env, "max_pipe_progress"):
        env.max_pipe_progress = current_progress.clone()
        
    # 2. 环境重置处理：如果环境刚好触发了 Reset，历史最大记录要清空为起点状态
    env.max_pipe_progress = torch.where(
        env.reset_buf | (current_progress < 0),  # 如果发生reset或是掉管(-1)，重置历史记录
        current_progress, 
        env.max_pipe_progress
    )
    # 3. 计算增量并截断（仅允许正向增长）
    delta = current_progress - env.max_pipe_progress
    reward = torch.clamp(delta, min=0.0)
    # 4. 更新历史最大里程
    env.max_pipe_progress = torch.max(env.max_pipe_progress, current_progress)
    # 相当于奖励是与历史最大值的差值
    return reward.view(env.num_envs)

def conditional_posture_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """
    越障姿态掩码：只有在夹持臂接触管道时，才惩罚姿态误差
    """
    # 获取欧拉角误差 [N, 2]
    pose_error = p_geom.get_target_relative_pose(env, asset_cfg) 
    err_sum = torch.sum(torch.abs(pose_error), dim=-1) # 各方向误差之和
    
    # 判断该侧是否和管道有接触（这里只要几个传感器接触力的总和大于阈值即认为“有接触”）
    # 直接隐式迭代求和避免多余维度的隐式 stack 创建降低性能
    total_force = sum(torch.norm(env.scene.sensors[cfg.name].data.net_forces_w, dim=-1).squeeze(-1) for cfg in sensor_cfgs)
    
    # 条件掩码：接触力大于阈值(如1.0N)视为处于抓管状态，执行惩罚；否则(悬空越障)返回0
    penalty = torch.where(total_force > 1.0, err_sum, torch.zeros_like(err_sum))
    return penalty.view(env.num_envs)

def wheel_contact_count_reward(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    threshold: float = 1.0,
    min_steps: int = 200,
    no_contact_value: float = -10.0,
) -> torch.Tensor:
    """
    按接触轮子数量给正向奖励：每个接触轮子 +1，6个轮子最高 +6。
    """
    contact_count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for cfg in sensor_cfgs:
        force_mag = torch.norm(env.scene.sensors[cfg.name].data.net_forces_w, dim=-1, p=2).view(env.num_envs)
        contact_count += (force_mag > threshold).to(torch.float32)

    # too young 阶段不触发“0接触重罚”，避免刚重置时落体阶段误导策略。
    too_young = env.episode_length_buf < min_steps
    no_contact = contact_count < 1.0
    contact_count = torch.where(
        torch.logical_and(~too_young, no_contact),
        torch.full_like(contact_count, no_contact_value),
        contact_count,
    )

    return contact_count

@configclass
class RewardsCfg:
    # -----------------------------
    # 1. 引导走向终点 (Task Progress)
    # -----------------------------
    # 最大里程奖励 (根据配置，取机器人前端 FM_24_link 前进数值最准确)
    progress_reward = RewTerm(
        func=axial_progress_reward,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])},
        weight=500.0  # 放大前进信号，提升每步有效梯度
    )
    
    # 时间流逝惩罚 (让其尽量少花时间)
    alive_penalty = RewTerm(
        func=mdp.is_alive,
        weight = 0.03  # 0409: 改为正值, -0.08
    )
    
    # 任务奖励：到达终点
    reach_target_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["reach_goal_reset"]}, 
        weight=300.0  # 终点奖励与放大后的过程奖励保持比例
    )
    
    # 任务惩罚：掉出管外（可选）
    fall_off_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["lost_all_contacts"]},
        weight=-200.0  # 保持失败惩罚显著，防止“快速掉落”策略
    )
    
    # 任务惩罚：卡死不动（可选）
    stagnation_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["stagnation_reset"]},
        weight=-200.0  # 惩罚卡死，鼓励
    )

    # -----------------------------
    # 2. 运动技能 (Motion Skills)
    # -----------------------------
    # 前夹持臂姿态对齐 (条件掩码)
    front_posture_alignment = RewTerm(
        func=conditional_posture_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"]),
            "sensor_cfgs": [SceneEntityCfg("touch_m2"), SceneEntityCfg("touch_a3"), SceneEntityCfg("touch_a4")] 
        },
        weight=-0.3
    )
    
    # 后夹持臂姿态对齐 (条件掩码)
    back_posture_alignment = RewTerm(
        func=conditional_posture_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"]),
            "sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")] 
        },
        weight=-0.3
    )

    # -----------------------------
    # 3. 运动稳定性正则化 (Stability & Regularization)
    # -----------------------------
    # 动作速率平滑度惩罚: 限制输出跳变
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01
    )
    
    # 关节加速度惩罚: 防止模型输出引发高频震荡
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7
    )
    
    # # 功耗/扭矩惩罚: 提倡多轮共同平缓发力
    # joint_torques_l2 = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-1e-7
    # )
    
    # 轮子接触数奖励: 每个接触轮子 +1，最多 +6
    wheel_contact_reward = RewTerm(
        func=wheel_contact_count_reward,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("touch_m1"),
                SceneEntityCfg("touch_m2"),
                SceneEntityCfg("touch_a1"),
                SceneEntityCfg("touch_a2"),
                SceneEntityCfg("touch_a3"),
                SceneEntityCfg("touch_a4"),
            ],
            "threshold": 1.0,
            "min_steps": 200,
            "no_contact_value": -6.0,
        },
        weight=2.0
    )