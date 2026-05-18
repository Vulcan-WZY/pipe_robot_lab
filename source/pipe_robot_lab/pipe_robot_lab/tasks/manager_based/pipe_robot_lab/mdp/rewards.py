# ===========
# Date: 2026-01-11 15:20
# Author: Vulcan
# LastEditTime: 2026-05-17 00:21
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


def _get_contact_count(env: ManagerBasedRLEnv, sensor_cfgs: list[SceneEntityCfg], threshold: float = 1.0) -> torch.Tensor:
    """统计一侧或全身接触轮的数量。"""
    contact_count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for cfg in sensor_cfgs:
        force_mag = torch.norm(env.scene.sensors[cfg.name].data.net_forces_w, dim=-1, p=2).view(env.num_envs)
        contact_count += (force_mag > threshold).to(torch.float32)
    return contact_count

def dia_matched_reward(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,       # 输入的单侧夹持臂机体(如 FM_24_link / BM_01_link)
    mid_arms: list[str],            # 需要参与匹配的 mid_arm 关节名称列表
    sensor_cfgs: list[SceneEntityCfg],  # 该侧对应的接触传感器列表
    sigma: float = 0.4,             # 误差容忍度 (弧度)
    min_contact: int = 2,           # 至少有几个轮子接触时才启用此奖励
    pre_contact_scale: float = 0.15,
) -> torch.Tensor:
    """
    夹持臂角度与管道直径的匹配奖励.
    
    1. 根据 body_cfg 指定的机体在世界坐标系下的位置, 查询其当前所在管道段的直径 D.
    2. 根据 MATLAB 拟合的六次多项式, 由 D 计算理论上最优的 mid_arm 夹持角度.
    3. 计算当前 mid_arm 关节角度与目标角度的误差, 通过高斯核函数映射为奖励值.
    4. 仅在该侧接触轮数 >= min_contact 时才给出奖励信号, 避免空中无意义梯度.
    """
    asset = env.scene[body_cfg.name]
    
    # 0. 接触数条件掩码
    contact_count = _get_contact_count(env, sensor_cfgs, threshold=1.0)
    contact_mask = contact_count >= min_contact
    
    # 1. 获取该侧机体所在管道的局部信息
    pipe_info = p_geom.get_cached_pipe_info(env, body_cfg)
    
    # 提取管段ID (整数部分)
    seg_idx = pipe_info[:, 3].floor().long()
    valid_mask_seg = seg_idx >= 0
    safe_seg_idx = torch.clamp(seg_idx, min=0)
    
    # 2. 从 pipe_info 获取对应管段的直径 D (单位: 米)
    D_meters = env._gpu_pipe_info[safe_seg_idx, 2]
    
    # 转换为多项式输入尺度: 直径 mm/100 -> dm (decimeters)
    dia = D_meters * 10.0
    
    # 3. 使用六次多项式计算理论最优角度 (单位: 度)
    c = [-3.1762, 38.3970, -184.8731, 453.9609, -593.4035, 408.7483, -85.6488]
    target_angle_deg = (
        c[0] * dia**6 +
        c[1] * dia**5 + 
        c[2] * dia**4 +
        c[3] * dia**3 +
        c[4] * dia**2 +
        c[5] * dia +
        c[6]
    )
    
    # 减去装配偏置 57.63 度, 并转为弧度
    target_angle_rad = torch.deg2rad(target_angle_deg - 57.63)
    
    # 4. 获取当前 mid_arm 关节的角度
    joint_ids = asset.find_joints(mid_arms)[0]
    current_angles = asset.data.joint_pos[:, joint_ids]
    
    # 将目标角度广播到所有 mid_arm
    target_angles_rad_expanded = target_angle_rad.unsqueeze(1).expand_as(current_angles)
    
    # 5. 计算 W(err) 奖励: 高斯核函数
    err = current_angles - target_angles_rad_expanded
    reward = torch.exp(- (err ** 2) / (sigma ** 2))
    
    # 对多个 mid_arm 取平均
    mean_reward = reward.mean(dim=1)
    
    # 如果没在有效管段上, 奖励为0
    mean_reward = torch.where(valid_mask_seg, mean_reward, torch.zeros_like(mean_reward))
    
    # 软门控: 接触前保留较弱引导，接触后给予完整奖励
    gate = torch.where(contact_mask, torch.ones_like(mean_reward), torch.full_like(mean_reward, pre_contact_scale))
    mean_reward = mean_reward * gate
    
    return mean_reward


def up_arm_pre_grasp_reward(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    joint_names: list[str],
    sensor_cfgs: list[SceneEntityCfg],
    threshold: float = 1.0,
    contact_stop_count: int = 1,
) -> torch.Tensor:
    """
    预夹持阶段的弱引导：在尚未形成有效接触前，鼓励 up_arm 朝更夹紧的方向运动。
    一旦接触形成，该奖励自动关闭，避免长期诱导策略无脑夹到极限。
    """
    asset = env.scene[body_cfg.name]
    joint_ids = asset.find_joints(joint_names)[0]
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, joint_ids]
    joint_min = joint_limits[..., 0]
    joint_max = joint_limits[..., 1]
    joint_span = torch.clamp(joint_max - joint_min, min=1e-6)

    # up_arm 越小越紧，将其映射到 [0, 1]
    closeness = torch.clamp((joint_max - joint_pos) / joint_span, min=0.0, max=1.0)
    mean_closeness = closeness.mean(dim=1)

    contact_count = _get_contact_count(env, sensor_cfgs, threshold=threshold)
    pre_contact_mask = contact_count < contact_stop_count
    return torch.where(pre_contact_mask, mean_closeness, torch.zeros_like(mean_closeness))


def steer_wheel_action_penalty(
    env: ManagerBasedRLEnv,
    action_indices: list[int],
) -> torch.Tensor:
    """
    惩罚舵轮动作维度的输出幅值, 用于在夹持课程阶段锁定轮子不动.
    action_indices: 舵轮动作在整个动作向量中的维度索引列表.
    """
    actions = env.action_manager.action
    wheel_actions = actions[:, action_indices]
    return torch.sum(wheel_actions ** 2, dim=-1)


def frozen_action_deviation_penalty(
    env: ManagerBasedRLEnv,
    action_indices: list[int],
) -> torch.Tensor:
    """
    课程阶段对冻结动作维度仍保留一个轻微惩罚，帮助 actor 逐步学会这些维度应输出接近 0。
    """
    actions = env.action_manager.action
    frozen_actions = actions[:, action_indices]
    return torch.sum(frozen_actions ** 2, dim=-1)


def bend_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
) -> torch.Tensor:
    """
    惩罚 bend 关节偏离初始零位的幅度, 用于在夹持课程阶段冻结弯折自由度.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset.find_joints(joint_names)[0]
    joint_pos = asset.data.joint_pos[:, joint_ids]
    return torch.sum(joint_pos ** 2, dim=-1)
    

@configclass
class RewardsCfg:
    # -----------------------------
    # 1. 引导走向终点 (Task Progress)
    # -----------------------------
    
    # 生存超时结算奖励 (当成功活完设定的最大限制步数时，获取一笔大额正向鼓励)
    survival_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["time_out"]},
        weight=0.0,
    )
    
    # 最大里程奖励 (根据配置，取机器人前端 FM_24_link 前进数值最准确)
    progress_reward = RewTerm(
        func=axial_progress_reward,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])},
        weight=0.0  # [改动] 初期不要求前进，设为0防止干扰夹持策略
    )
    
    # 时间流逝惩罚 (让其尽量少花时间)
    alive_penalty = RewTerm(
        func=mdp.is_alive,
        weight = 0.03  # 活得越久分越高，与 survival_bonus 一起鼓励不掉落
    )
    
    # 任务奖励：到达终点
    reach_target_bonus = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["reach_goal_reset"]}, 
        weight=0.0  # [改动] 初期不用管终点
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
    # front_posture_alignment = RewTerm(
    #     func=conditional_posture_penalty,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"]),
    #         "sensor_cfgs": [SceneEntityCfg("touch_m2"), SceneEntityCfg("touch_a3"), SceneEntityCfg("touch_a4")] 
    #     },
    #     weight=-0.3
    # )
    
    # # 后夹持臂姿态对齐 (条件掩码)
    # back_posture_alignment = RewTerm(
    #     func=conditional_posture_penalty,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"]),
    #         "sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")] 
    #     },
    #     weight=-0.3
    # )
    
    # 前侧夹持臂角度匹配奖励 (基于 FM_24_link 位置查询管道直径)
    front_dia_matched = RewTerm(
        func=dia_matched_reward,
        params={
            "body_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"]),
            "mid_arms": ["mid_arm_03", "mid_arm_04"],
            "sensor_cfgs": [SceneEntityCfg("touch_m2"), SceneEntityCfg("touch_a3"), SceneEntityCfg("touch_a4")],
            "sigma": 0.45,
            "min_contact": 2,
            "pre_contact_scale": 0.15,
        },
        weight=6.0
    )

    back_dia_matched = RewTerm(
        func=dia_matched_reward,
        params={
            "body_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"]),
            "mid_arms": ["mid_arm_01", "mid_arm_02"],
            "sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")],
            "sigma": 0.45,
            "min_contact": 2,
            "pre_contact_scale": 0.15,
        },
        weight=6.0
    )

    front_up_arm_pre_grasp = RewTerm(
        func=up_arm_pre_grasp_reward,
        params={
            "body_cfg": SceneEntityCfg("robot"),
            "joint_names": ["up_arm_03", "up_arm_04"],
            "sensor_cfgs": [SceneEntityCfg("touch_m2"), SceneEntityCfg("touch_a3"), SceneEntityCfg("touch_a4")],
            "threshold": 1.0,
            "contact_stop_count": 1,
        },
        weight=0.25
    )

    back_up_arm_pre_grasp = RewTerm(
        func=up_arm_pre_grasp_reward,
        params={
            "body_cfg": SceneEntityCfg("robot"),
            "joint_names": ["up_arm_01", "up_arm_02"],
            "sensor_cfgs": [SceneEntityCfg("touch_m1"), SceneEntityCfg("touch_a1"), SceneEntityCfg("touch_a2")],
            "threshold": 1.0,
            "contact_stop_count": 1,
        },
        weight=0.25
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
        weight=-5e-8
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
            "min_steps": 100,
            "no_contact_value": -6.0,
        },
        weight=0.75
    )
    
    # 舵轮动作惩罚: 夹持课程阶段锁定轮子不输出动作
    steer_wheel_lock = RewTerm(
        func=frozen_action_deviation_penalty,
        params={
            "action_indices": list(range(0, 12)),
        },
        weight=-0.25
    )
    
    # 弯折偏离惩罚: 夹持课程阶段冻结 bend 自由度
    bend_lock = RewTerm(
        func=bend_deviation_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ["bend_01", "bend_02"],
        },
        weight=-0.25
    )
