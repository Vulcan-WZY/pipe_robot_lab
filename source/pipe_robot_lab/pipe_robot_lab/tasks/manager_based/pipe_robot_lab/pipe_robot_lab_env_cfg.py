# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg , ImuCfg

##
# Pre-defined configs
##
from pipe_robot_lab.assets.pipe_robot.pipe_robot_cfg import PIPE_ROBOT_CFG  # isort:skip
from .mdp.actions import ActionsCfg  
from .mdp.observations import ObservationsCfg
from .mdp.rewards import RewardsCfg
from .mdp.events import EventCfg
from .mdp.terminations import TerminationsCfg

# ! 引入随机加载管道
import os
import random
import glob
# 定位到生成器产生的管道路径
from pipe_robot_lab.assets.pipe_env.pipe_generator import CURRENT_DIR
MESHES_DIR = os.path.join(CURRENT_DIR, "usd")
JSON_DIR = os.path.join(CURRENT_DIR, "json")

# 获取所有生成的 STL 文件 (排除模板文件 stand_pipe.STL)
all_pipe_stls = [p for p in glob.glob(os.path.join(MESHES_DIR, "*.usd"))]
# 随机选择一个管道 (如果文件夹为空则给个默认空字符串防报错)
SELECTED_PIPE_STL = random.choice(all_pipe_stls) if all_pipe_stls else ""
# 推导出对应的 JSON 文件路径
SELECTED_PIPE_JSON = SELECTED_PIPE_STL.replace("usd", "json").replace(".usd", ".json") if SELECTED_PIPE_STL else ""
print(f"[INFO] Selected Pipe Environment: {SELECTED_PIPE_STL}")
##
# Scene definition
##
@configclass
class PipeRobotSceneCfg(InteractiveSceneCfg):
    # 地面
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane", 
    #     spawn=sim_utils.GroundPlaneCfg()
    # )
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/lightDistant", 
        spawn=sim_utils.DistantLightCfg(intensity=5000.0)
    )
    
    # * 改为使用随机生成的管道
    pipe_env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PipeEnv",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SELECTED_PIPE_STL,
            rigid_props= None,
            collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,  # 增加接触偏移以改善碰撞检测
                    rest_offset=0.001,      # 设置休息偏移为0以确保精确碰撞响应
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # 根据生成器设置初始位置
            rot=(1.0, 0.0, 0.0, 0.0), # 默认无旋转 (w, x, y, z)
        ),
    )
    
    # # 管道障碍物 (静态)
    # pipe_obstacle01 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/PipeObstacle1",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.18,        # 直径 360mm
    #         height=2.0,         # 长度 2m
    #         rigid_props=None,   # 静态 (Static)
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         # 配置高摩擦力材质
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0,   # 静摩擦系数
    #             dynamic_friction=1.0,  # 动摩擦系数
    #             restitution=0.0,       # 恢复系数(0表示不反弹)
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.5), # 中心位置
    #         rot=(0.70711, 0.70711, 0.0, 0.0), # 沿Y轴放置 (绕X轴转90度)
    #     ),
    # )
    # pipe_obstacle02 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/PipeObstacle2",
    #     # spawn= si
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.18,        # 直径 360mm
    #         height=2.0,         # 长度 2m
    #         rigid_props=None,   # 静态 (Static)
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         # 配置高摩擦力材质
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0,   # 静摩擦系数
    #             dynamic_friction=1.0,  # 动摩擦系数
    #             restitution=0.0,       # 恢复系数(0表示不反弹)
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 1.0, 1.5), # 中心位置
    #         rot=(1, 0.0, 0.0, 0.0), # 沿Y轴放置 (绕X轴转90度)
    #     ),
    # )
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

##
# Environment configuration
##

@configclass
class PipeRobotLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PipeRobotSceneCfg = PipeRobotSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        
        self.sim.physx.bounce_threshold_velocity = 0.2