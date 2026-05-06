import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg , ImuCfg , ContactSensorCfg

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
import json
import yaml
import torch
# 定位到生成器产生的管道路径
from pipe_robot_lab.assets.pipe_env.pipe_generator import CURRENT_DIR
MESHES_DIR = os.path.join(CURRENT_DIR, "usd")
JSON_DIR = os.path.join(CURRENT_DIR, "json")

# 读取 auto_train.yaml 中配置的区间以过滤管道 ID (动态课程学习支持)
AUTO_TRAIN_YAML = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../sh/config/auto_train.yaml"))
filter_range = None
if os.path.exists(AUTO_TRAIN_YAML):
    try:
        with open(AUTO_TRAIN_YAML, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            filter_range = cfg.get("training", {}).get("env_pipe_id_range", None)
    except Exception as e:
        print(f"[Warning] Could not parse pipe ID range from auto_train.yaml: {e}")

# 获取所有生成的 USD 文件 (排除模板文件 stand_pipe), 并按照区间过滤
all_pipe_stls = []
for p in glob.glob(os.path.join(MESHES_DIR, "*.usd")):
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem.isdigit():
        obj_id = int(stem)
        if filter_range is not None and len(filter_range) == 2:
            if filter_range[0] <= obj_id <= filter_range[1]:
                all_pipe_stls.append(p)
        else:
            all_pipe_stls.append(p)

# 随机选择一个管道 (如果文件夹为空则给个默认空字符串防报错)
SELECTED_PIPE_STL = random.choice(all_pipe_stls) if all_pipe_stls else ""
# 推导出对应的 JSON 文件路径
SELECTED_PIPE_JSON = SELECTED_PIPE_STL.replace("usd", "json").replace(".usd", ".json") if SELECTED_PIPE_STL else ""
# * 根据选取的管路中第一节管道直径得到机器人加载位置偏置
ROBOT_INIT_Z = 0.5 # 先给默认值
if os.path.exists(SELECTED_PIPE_JSON):
    with open(SELECTED_PIPE_JSON, 'r',encoding='utf-8') as f:
        _temp_data = json.load(f)
        pipe_dia = _temp_data[0]["info"][2]
        pipe_radius = pipe_dia / 2.0

        bias = 0.031 + 0.005 # 机器人模型原点到主驱动轮下端面距离, 并给予额外的安全间隙
        ROBOT_INIT_Z = pipe_radius + bias
        print(f"[INFO] Selected Pipe: {os.path.basename(SELECTED_PIPE_STL)} | Dia: {pipe_dia:.3f} | Set Robot Initial Z to: {ROBOT_INIT_Z:.3f}m")
# ! 随机管道代码段结束

# ! 260504 课程学习框架相关 引入随机管道直径开关
USE_USD_PIPE = True # 使用USD文件集中的随机管道文件, 如果改为false则使用一段随机直径的直管道
# ! end

##
# Scene definition
##
@configclass
class PipeRobotSceneCfg(InteractiveSceneCfg):
    # 地面 (修改: 添加 color 属性以生成纯色地面, 避免试图从AWS下载默认网格USD导致网络报错)
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # 光照
    light = AssetBaseCfg(
        prim_path="/World/lightDistant", 
        spawn=sim_utils.DistantLightCfg(intensity=2500.0)
    )
    
    # 天空布景
    sky = AssetBaseCfg(
        prim_path="/World/skyDome",
        spawn=sim_utils.DomeLightCfg(
            # 基础环境光强度
            intensity=900.0,
            # 【方案A：纯色天空】赋予全局光一个天蓝色调。如果网络不好加载不了贴图，这个最实用且效果很好。
            color=(0.75, 0.85, 1.0),
            # 【方案B：真实的 HDR 贴图天空】如果网络能连接 NVIDIA 默认服务器，取消下面这行注释。
            # 官方最常用的无云/少云晴空 HDR，加上后会有真实的太阳光晕和天空渐变。
            # texture_file="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/NVIDIA/Backgrounds/skies/clear_sky.hdr",
        )
    )
    
    # * 改为使用随机生成的管道
    pipe_env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PipeEnv",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SELECTED_PIPE_STL,
            rigid_props= None,
            collision_props=sim_utils.CollisionPropertiesCfg(
                    # 保证 contact_offset > rest_offset，避免接触生成过晚导致轮子“插入”管壁
                    contact_offset=0.03,
                    rest_offset=0.0,
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
    # 机器人 (必须使用 {ENV_REGEX_NS} 占位符)
    robot: ArticulationCfg = PIPE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.1, ROBOT_INIT_Z) # 根据管道直径动态设置初始Z高度
    
    back_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BM_02_link_cam/cam_back",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)), # 绕 z 轴旋转 180 度修正倒立 (w, x, y, z)
        update_period=0.1,  # 10Hz
        height=120,
        width=160,
        data_types=["depth"],  # Intel D435i 深度流
        debug_vis=False, # 会导致 IsaacLab 每一帧都把带深度图结果的 UI viewport 画出来并强制渲染到主窗口上
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
        height=120,
        width=160,
        data_types=["depth"],  # Intel D435i 深度流
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,          # 对应 D435i 约 1.93mm 焦距
            horizontal_aperture=3.8,    # 对应约 86° 水平 FOV
            vertical_aperture=2.39,     # 对应约 57° 垂直 FOV
            clipping_range=(0.1, 10.0), # D435i 有效深度范围
        ),
    )
    # back_imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/BM_03_link_imu",
    #     update_period=0.005,           # 200Hz
    # )
    # front_imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/FM_26_link_imu",
    #     update_period=0.005,           # 200Hz
    # )

    # * 定义一组接触传感器
    # 后侧主动轮
    touch_m1 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/BM_08_link",
        update_period= 0.02,
    )
    # 前侧主动轮
    touch_m2 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/FM_31_link",
        update_period= 0.02,
    )
    # 后侧左辅助轮
    touch_a1 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/BL_22_link",
        update_period= 0.02,
    )
    # 后侧右辅助轮
    touch_a2 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/BR_23_link",
        update_period= 0.02,
    )
    # 前侧左辅助轮
    touch_a3 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/FL_44_link",
        update_period= 0.02,
    )
    # 前侧右辅助轮
    touch_a4 = ContactSensorCfg(
        prim_path= "{ENV_REGEX_NS}/Robot/FL_45_link",  # 这里竟然在USD中的名字是FL我难绷
        update_period= 0.02,
    )
##
# Environment configuration
##

@configclass
class PipeRobotLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PipeRobotSceneCfg = PipeRobotSceneCfg(num_envs = 1, env_spacing = 4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 新增字段：用于存储当前管道的详细参数 (在__post_init__中动态挂载，不写类型注解防止OmegaConf试图解析Tensor引发崩溃)
    # 本来是: pipe_transform: torch.Tensor = None
    # 本来是: pipe_info: torch.Tensor = None
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.decimation = 2                         # 抽帧倍数 (控制渲染频率，降低CPU/GPU负载), 
        # 同时也决定了Agent的控制频率 (control_freq = sim_freq / decimation = 100 / 2 = 50Hz)
        self.sim.dt = 1 / 100                       # 物理仿真时间步长(底层物理引擎每0.01秒更新一次)
        self.episode_length_s = 12.0 # 12秒 * 50Hz (控制步频) = 恰好600步
        self.sim.render_interval = self.decimation
        
        self.sim.physx.bounce_threshold_velocity = 0.2
    
        # * 读取并挂载本轮随机加载的JSON管道描述文件
        if os.path.exists(SELECTED_PIPE_JSON):
            with open(SELECTED_PIPE_JSON, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            transform_list = [item["transform"] for item in raw_data]
            info_list = [item["info"] for item in raw_data]
            
            # 注意: OmegaConf 不支持 Tensor 作为配置值，因此这里存储为可序列化的 list。
            transform_tensor = torch.tensor(transform_list, dtype=torch.float32)  # [N, 4, 4]
            info_tensor = torch.tensor(info_list, dtype=torch.float32)            # [N, 6]
            transform_inv_tensor = torch.linalg.inv(transform_tensor)

            self.pipe_transform = transform_tensor.tolist()
            self.pipe_info = info_tensor.tolist()
            self.pipe_transform_inv = transform_inv_tensor.tolist()
            print(f"[INFO] Pipe Transform Tensor Shape: {transform_tensor.shape}")
            print(f"[INFO] Pipe Info Tensor Shape: {info_tensor.shape}")
        else:
            default_transform = torch.eye(4).unsqueeze(0)  # Shape: [1, 4, 4]
            default_info = torch.tensor([[0, 0, 0.36, 1.0, 0.0, 0.0]], dtype=torch.float32) # Shape: [1, 6]
            self.pipe_transform = default_transform.tolist()
            self.pipe_info = default_info.tolist()
            self.pipe_transform_inv = torch.linalg.inv(default_transform).tolist()
            print(f"[WARNING] Could not find JSON info at: {SELECTED_PIPE_JSON}")
        
        # 将 DebugCfg 注册进环境以便测试框架拾取
        # self.observations.debug = ObservationsCfg.DebugCfg()
        