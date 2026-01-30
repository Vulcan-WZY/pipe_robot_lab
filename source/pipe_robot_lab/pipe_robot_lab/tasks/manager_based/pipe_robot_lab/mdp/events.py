from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
# =============================================================================
# 5. 事件配置 (Events Configuration)
# =============================================================================
@configclass
class EventCfg:
    # Reset 时重置关节位置到默认附近 (微小随机)
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    