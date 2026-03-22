# ===========
# Date: 2026-01-11 15:20
# Author: Vulcan
# LastEditTime: 2026-03-22 19:32
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
    forces = [torch.norm(env.scene.sensors[cfg.name].data.net_forces_w, dim=-1) for cfg in sensor_cfgs]
    total_force = torch.sum(torch.stack(forces, dim=0), dim=0)
    
    # 条件掩码：接触力大于阈值(如1.0N)视为处于抓管状态，执行惩罚；否则(悬空越障)返回0
    penalty = torch.where(total_force > 1.0, err_sum, torch.zeros_like(err_sum))
    return penalty.view(env.num_envs)

def critical_contact_penalty(env: ManagerBasedRLEnv, front_sensor_cfgs: list[SceneEntityCfg], back_sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """
    关键支撑基线约束：任何情况下，首尾两端至少有一端必须保持完整的三点接触，否则重罚
    """
    def is_three_point_contact(cfgs):
        # 判定给定的一组（3个）传感器是否全部产生接触受力
        contacts = [(torch.norm(env.scene.sensors[c.name].data.net_forces_w, dim=-1) > 1.0) for c in cfgs]
        return contacts[0] & contacts[1] & contacts[2]

    front_secure = is_three_point_contact(front_sensor_cfgs)
    back_secure = is_three_point_contact(back_sensor_cfgs)
    
    # 只要有一端是稳定的三点连接，即认为是安全的平衡态
    is_safe = front_secure | back_secure
    
    # 返回一个 0.0 或 1.0 的信号张量，为 1.0 时代表触发惩罚
    penalty = torch.where(is_safe, torch.zeros_like(is_safe, dtype=torch.float), torch.ones_like(is_safe, dtype=torch.float))
    return penalty.view(env.num_envs)

@configclass
class RewardsCfg:
    # -----------------------------
    # 1. 引导走向终点 (Task Progress)
    # -----------------------------
    # 最大里程奖励 (根据配置，取机器人前端 FM_24_link 前进数值最准确)
    progress_reward = RewTerm(
        func=axial_progress_reward,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])},
        weight=100.0  # 假设总管段长度为10节，每走一节给100分，可自行缩放
    )
    
    # 时间流逝惩罚 (让其尽量少花时间)
    alive_penalty = RewTerm(
        func=mdp.is_alive,
        weight=-0.05
    )
    
    # 任务奖励：到达终点
    reach_target_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["reach_goal_reset"]}, 
        weight=200.0  # 给予极大的奖励（通关！）
    )
    
    # 任务惩罚：掉出管外（可选）
    fall_off_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["lost_all_contacts"]},
        weight=-50.0  # 如果是因为掉出管外导致重启的，给与重重惩罚
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
        weight=-2.0
    )
    
    # 后夹持臂姿态对齐 (条件掩码)
    back_posture_alignment = RewTerm(
        func=conditional_posture_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"]),
            "sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")] 
        },
        weight=-2.0
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
        weight=-2e-5
    )
    
    # 功耗/扭矩惩罚: 提倡多轮共同平缓发力
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5
    )
    
    # 基础支撑底线约束: 只要有一侧不是完整的3点接触，就持续扣分
    critical_contact_loss = RewTerm(
        func=critical_contact_penalty,
        params={
            "front_sensor_cfgs": [SceneEntityCfg("touch_m2"), SceneEntityCfg("touch_a3"), SceneEntityCfg("touch_a4")],
            "back_sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")] 
        },
        weight=-1.5
    )