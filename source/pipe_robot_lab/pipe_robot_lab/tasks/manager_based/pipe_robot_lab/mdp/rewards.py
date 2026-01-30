import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
# =============================================================================
# 6. 其他 RL 必需配置 (Rewards, Terminations)
# =============================================================================
@configclass
class RewardsCfg:
    """空奖励配置 (Demo用途)"""
    pass