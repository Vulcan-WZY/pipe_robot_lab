import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

from pipe_robot_action import ActionsCfg
from pipe_robot_obs import ObservationsCfg
from pipe_robot_reward import RewardsCfg
from pipe_robot_events import EventCfg, TerminationsCfg
# 用于挂载相机与IMU
from isaaclab.sensors import TiledCameraCfg , ImuCfg
import os
# =============================================================================
# 1. 机器人资产配置 (Robot Asset Configuration)
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# USD_PATH = os.path.join(CURRENT_DIR, "../source/model/pipe_robot/USD/pipe_robot_rename/pipe_robot_rename.usd")
USD_PATH = os.path.join(CURRENT_DIR, "../source/pipe_robot_lab/pipe_robot_lab/assets/pipe_robot/usd/pipe_robot_rename.usd")
    
PIPE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            r".*main_steer_.*": 0.0,  # 所有主动轮舵向
            r".*main_wheel_.*": 0.0,  # 所有主动轮轮速
            r".*up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r".*mid_arm_.*": 0.8,     # 所有中臂关节(大臂与小臂连接点)
            r".*tail_arm_.*": 0.0,    # 所有尾臂关节(小臂与末端辅助轮连接点)
            r".*assist_steer_.*": 0.0, # 所有辅助轮舵向
            r".*assist_wheel_.*": 0.0, # 所有辅助轮轮速
            r".*bend_.*": 0.0,         # 所有弯折变形关节
        },
        joint_vel={
            r".*": 0.0,
        }
    ),
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=[".*main_steer_.*", ".*assist_steer_.*"],
            effort_limit_sim={
                ".*main_steer_.*":    100.0,
                ".*assist_steer_.*":  100.0,
            },
            velocity_limit_sim=     5.0,
            stiffness=              5000.0, # 提高刚度确保位置准确
            damping=                100.0,
            # effort_limit_sim=.0,
        ),
        "wheel": ImplicitActuatorCfg( # 轮子: 适合速度控制 (Stiffness=0)
            joint_names_expr=[".*main_wheel_.*",".*assist_wheel_.*"],
            effort_limit_sim={
                ".*main_wheel_.*":    100.0,
                ".*assist_wheel_.*":  100.0,
            },
            velocity_limit_sim=     20.0,
            stiffness=              0.0,
            damping=                10.0,
            # effort_limit_sim=50.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*up_arm_.*",".*mid_arm_.*",".*tail_arm_.*"],
            effort_limit_sim={
                ".*up_arm_.*":    2000.0,
                ".*mid_arm_.*":   2000.0,
                ".*tail_arm_.*":  5000.0,
            },
            velocity_limit_sim= {
                ".*up_arm_.*":    10.0,
                ".*mid_arm_.*":   10.0,
                ".*tail_arm_.*":  20.0,
            },
            stiffness=          5000.0,
            damping=            100.0,
        ),
        "bend": ImplicitActuatorCfg(
            joint_names_expr=[".*bend_.*"],
            effort_limit_sim=   500.0,
            velocity_limit_sim= 10.0,
            stiffness=          1000.0,
            damping=            100.0,
        )
    },

)


# =============================================================================
# 2. 场景配置 (Scene Configuration)
# =============================================================================
@configclass
class PipeRobotSceneCfg(InteractiveSceneCfg):
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg()
    )
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/lightDistant", 
        spawn=sim_utils.DistantLightCfg(intensity=5000.0)
    )
    # 管道障碍物 (静态)
    pipe_obstacle01 = AssetBaseCfg(
        prim_path="/World/PipeObstacle1",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,        # 直径 360mm
            height=2.0,         # 长度 2m
            rigid_props=None,   # 静态 (Static)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 配置高摩擦力材质
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,   # 静摩擦系数
                dynamic_friction=1.0,  # 动摩擦系数
                restitution=0.0,       # 恢复系数(0表示不反弹)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), # 中心位置
            rot=(0.70711, 0.70711, 0.0, 0.0), # 沿Y轴放置 (绕X轴转90度)
        ),
    )
    pipe_obstacle02 = AssetBaseCfg(
        prim_path="/World/PipeObstacle2",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,        # 直径 360mm
            height=2.0,         # 长度 2m
            rigid_props=None,   # 静态 (Static)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 配置高摩擦力材质
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,   # 静摩擦系数
                dynamic_friction=1.0,  # 动摩擦系数
                restitution=0.0,       # 恢复系数(0表示不反弹)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.0, 1.5), # 中心位置
            rot=(1, 0.0, 0.0, 0.0), # 沿Y轴放置 (绕X轴转90度)
        ),
    )
    # 机器人 (必须使用 {ENV_REGEX_NS} 占位符)
    robot: ArticulationCfg = PIPE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    back_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BM_02_link_cam/cam_back",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)), # 绕 z 轴旋转 180 度修正倒立 (w, x, y, z)
        update_period=0.1,  # 10Hz
        height=240,
        width=320,
        data_types=["depth"],  # Intel D435i 深度流
        debug_vis=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,          # 对应 D435i 约 1.93mm 焦距
            horizontal_aperture=3.8,    # 对应约 86° 水平 FOV
            vertical_aperture=2.39,     # 对应约 57° 垂直 FOV
            clipping_range=(0.1, 10.0), # D435i 有效深度范围
        ),
    )
    front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FM_25_link_cam/cam_front",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)), # 绕 z 轴旋转 180 度修正倒立 (w, x, y, z)
        update_period=0.1,  # 10Hz
        height=240,
        width=320,
        data_types=["depth"],  # Intel D435i 深度流
        debug_vis=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,          # 对应 D435i 约 1.93mm 焦距
            horizontal_aperture=3.8,    # 对应约 86° 水平 FOV
            vertical_aperture=2.39,     # 对应约 57° 垂直 FOV
            clipping_range=(0.1, 10.0), # D435i 有效深度范围
        ),
    )
    back_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BM_03_link_imu",
        update_period=0.005,           # 200Hz
    )
    front_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FM_26_link_imu",
        update_period=0.005,           # 200Hz
    )


# =============================================================================
# 7. 环境总配置 (Environment Configuration)
# =============================================================================
@configclass
class PipeRobotEnvCfg(ManagerBasedRLEnvCfg):
    # 场景
    scene: PipeRobotSceneCfg = PipeRobotSceneCfg(num_envs=1, env_spacing=2.0)
    # 观测
    observations: ObservationsCfg = ObservationsCfg()
    # 动作
    actions: ActionsCfg = ActionsCfg()
    # 事件
    events: EventCfg = EventCfg()
    # 奖励 (RL必须)
    rewards: RewardsCfg = RewardsCfg()
    # 终止条件 (RL必须)
    terminations: TerminationsCfg = TerminationsCfg()
    # 回合长度 (RL必须)
    episode_length_s: float = 1000.0
    
    # 仿真参数
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.01, # 物理步长 10ms
        render_interval = 1, # 这里最好和相机帧率对应
        # gravity=(0.0, 0.0, 0.0),
        gravity=(0.0, 0.0, -9.81),
    )
    decimation = 1 # 决策频率 = 物理频率