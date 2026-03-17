# ===========
# Date: 2026-01-11 15:20
# Author: Vulcan
# LastEditTime: 2026-03-17 15:12
# Description: 主要配置管道检测机器人运动时的reward
# ==========
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import RecorderManager
# =============================================================================
# 6. 其他 RL 必需配置 (Rewards, Terminations)
# =============================================================================
@configclass
class RewardsCfg:
    # 任务奖励：到达终点
    reach_target_bonus = RewTerm(
        # IsaacLab 内置方法，捕获特定的 termination term
        func=mdp.is_terminated_term,
        # 当名为 "reach_goal_reset" 的终止条件被触发(为True)时，给对应环境分配一次性奖励
        params={"term_keys": ["reach_goal_reset"]}, 
        weight=200.0  # 给予极大的奖励（通关！）
    )
    
    # 任务惩罚：掉出管外（可选）
    fall_off_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["lost_all_contacts"]},
        weight=-50.0  # 如果是因为掉出管外导致重启的，给与重重惩罚
    )