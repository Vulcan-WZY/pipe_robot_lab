from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
# =============================================================================
# 5. 事件配置 (Events Configuration)
# =============================================================================
@configclass
class EventCfg:
    # 1. Reset 时重置机器人的基座(Root)位姿和速度到配置(CFG)中的默认值
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 范围全为0表示不增加任何随机偏移，严格从 CFG 中读取默认设定的 pos 和 rot
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # 2. Reset 时严格重置关节位置到默认处
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), # 1.0 表示严格等于机器人的默认关节角度
            "velocity_range": (0.0, 0.0), # 把重置时的关节速度强制清零
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    