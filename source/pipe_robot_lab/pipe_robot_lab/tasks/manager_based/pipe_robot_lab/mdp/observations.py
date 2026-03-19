from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
import torch


import pipe_robot_lab.funcs.pipe_geom_utils as p_geom

def get_depth_image(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, normalize: bool = True) -> torch.Tensor:
    """
    自定义函数：从 TiledCamera 中提取深度图
    """
    # 1. 根据配置名称获取传感器对象
    # env.scene.sensors 是一个字典，key 是你在 SceneCfg 中定义的变量名
    sensor = env.scene.sensors[sensor_cfg.name]
    # 2. 获取深度数据
    # TiledCamera 的数据结构通常是 sensor.data.output["depth"]
    # 形状通常是 [num_envs, height, width, 1]
    depth_data = sensor.data.output["depth"]
    # 3. 处理数据
    # 深度图可能包含无穷大 (Inf)，需要截断
    # 假设有效范围是 0.1m - 10.0m (参考你的 clipping_range)
    depth_data = torch.clamp(depth_data, min=0.0, max=10.0)
    
    # 4. 归一化 (可选，建议做) -> 映射到 [0, 1]
    if normalize:
        depth_data = depth_data / 10.0
        
    # 5. 调整维度
    # PyTorch 的 CNN 通常需要 [N, C, H, W]
    # 当前是 [N, H, W, 1]，我们需要 permute 成 [N, 1, H, W]
    # 或者对于有些 RL 库可能需要 flatten，但 SKRL 能够处理多维 Tensor
    depth_data = depth_data.permute(0, 3, 1, 2)
    
    return depth_data.clone() # 返回这部分数据的拷贝

def get_imu_orientation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read IMU orientation (quat_w) from sensor."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.quat_w.view(env.num_envs, -1)

def get_imu_ang_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read IMU angular velocity (ang_vel_b) from sensor."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ang_vel_b.view(env.num_envs, -1)

def get_imu_lin_acc(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read IMU linear acceleration (lin_acc_b) from sensor."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.lin_acc_b.view(env.num_envs, -1)
    
def get_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """ 返回接触力大小 """
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w # 常见字段：net_forces_w -> [num_envs, 3]
    return torch.norm(f, dim=-1, p=2)

def body_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """ 仅返回刚体在世界坐标系下的四元数姿态(wxyz) """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    return quat.reshape(env.num_envs, -1)



# ! ================================== 以下内容都是Reward中可能用到的，只在Debug中显示测试 ==================
def get_pose_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 1. 解析目标资产的位姿信息 -> [num_envs , 7] (x, y, z, qw, qx, qy, qz)
    poses = body_quat_w(env, asset_cfg) # 如果后续需要完整的x,y,z请结合mdp.body_pos或者从asset直接取完整pose
    # 但是 body_quat_w 只有姿态，我们需要完整的 7维 数据:
    
    # 3. 调用算子获取角度误差 (roll, yaw 偏差) -> [num_envs, 2]
    # 已使用懒加载缓存，直接传入 env 和 asset_cfg 即可
    error = p_geom.get_target_relative_pose(env, asset_cfg)
    return error

# =============================================================================
# 4. 观测配置 (Observations Configuration)
# =============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # # todo 数值空间观测
        # 观测所有主动轮和辅助轮的舵角方向
        steer_pos = ObsTerm(
            func = mdp.joint_pos, # 关节归一化位置
            params= {
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[".*main_steer_.*", ".*assist_steer_.*"]
                )            
            }
        )
        # 观测所有主动轮和辅助轮的轮速
        wheel_vel = ObsTerm(
            func= mdp.joint_vel, # 先使用真实速度
            params= {
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[".*main_steer_.*", ".*assist_steer_.*"]
                )            
            }
        )
        # # 观测变形推杆位置
        bend_pos = ObsTerm(
            func = mdp.joint_pos, # 关节位置
            params= {
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[".*bend_.*"]
                )            
            }
        )
        # 观测所有小臂位置
        arm_pos = ObsTerm(
            func = mdp.joint_pos_limit_normalized, # 关节归一化位置
            params= {
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[".*mid_arm_.*"]
                )            
            }
        )
        # 观测所有大臂up_arm力矩
        arm_torque = ObsTerm(
            func = mdp.joint_effort, # 关节归一化力矩
            params= {
                "asset_cfg": SceneEntityCfg("robot",
                    joint_names=[".*up_arm_.*"]
                )            
            }
        )
        # ! 暂时弃用IMU了，感觉可以直接从mdp中拿
        # # # todo IMU四元数姿态
        # imu_back_quat = ObsTerm(
        #     func = get_imu_orientation,
        #     params = {
        #         "sensor_cfg": SceneEntityCfg("back_imu")
        #     }
        # )
        # imu_front_quat = ObsTerm(
        #     func = get_imu_orientation,
        #     params = {
        #         "sensor_cfg": SceneEntityCfg("front_imu")
        #     }
        # )
        # # # todo IMU线加速度
        # imu_back_accel = ObsTerm(
        #     func = get_imu_lin_acc,
        #     params = {
        #         "sensor_cfg": SceneEntityCfg("back_imu")
        #     }
        # )
        # imu_front_accel = ObsTerm(
        #     func = get_imu_lin_acc,
        #     params = {
        #         "sensor_cfg": SceneEntityCfg("front_imu")
        #     }
        # )
        # 后部刚体位姿 (世界系): [pos(3), quat(4)]
        back_body_pose = ObsTerm(
            func= body_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"])}
        )
        # 前部刚体位姿 (世界系): [pos(3), quat(4)]
        front_body_pose = ObsTerm(
            func= body_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])}
        )
        # * 重力投影
        projected_gravity = ObsTerm(
            func = mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"])}
        )
        
        # * 轮子接触力
        touch_m1 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_m1")}
        )
        touch_m2 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_m2")}
        )
        touch_a1 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_a1")}
        )
        touch_a2 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_a2")}
        )
        touch_a3 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_a3")}
        )
        touch_a4 = ObsTerm(
            func= get_contact_force,
            params = {"sensor_cfg": SceneEntityCfg("touch_a4")}
        )
        # # todo 上一帧动作
        # last_action = ObsTerm(
        #     func = mdp.last_action,
        # )
        
        def __post_init__(self):
            # 计算观测维度
            # 拼接观测
            # self.concatenate_dim = True  # 将所有观测拼接成一个大向量
            self.enable_corruption = True # 启用观测扰动（如噪声）以增强鲁棒性 
            self.concatenate_terms = True
        
    @configclass
    class CameraCfg(ObsGroup):
        """视觉观测组：处理前后深度相机"""
        # 1. 后置相机 (back_cam)
        depth_back = ObsTerm(
            func=get_depth_image, # 调用上面的自定义函数
            params={
                "sensor_cfg": SceneEntityCfg("back_cam"), # 对应 SceneCfg 中的变量名
                "normalize": True
            }
        )
        # 2. 前置相机 (front_cam)
        depth_front = ObsTerm(
            func=get_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_cam"),
                "normalize": True
            }
        )
        
        def __post_init__(self):
            # 【关键】图像数据绝对不能拼接 (Concatenate)
            # 因为它们是多维 Tensor [N, 1, H, W]，而普通的 ObsTerm 都是 [N, D]
            # 设为 False 后，Isaac Lab 会将这个组以字典形式返回：
            # {"depth_back": Tensor, "depth_front": Tensor}
            self.concatenate_terms = False 
            # 是否开启图像噪声 (Sim-to-Real 时需要在 get_depth_image 里自己实现噪声逻辑，或开启 enable_corruption 并提供 noise model)
            # 对于图像，简单的加性噪声可能不够，通常暂设为 False 或自定义
            self.enable_corruption = False

        
        
    @configclass
    class CriticCfg(PolicyCfg):
        # Critic 观测：关节位置、速度和末端位姿
        back_link_vel = ObsTerm(
            func = mdp.base_lin_vel,
            params = {
                "asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"])
            }
        )  
        front_link_vel = ObsTerm(
            func = mdp.base_lin_vel,
            params = {
                "asset_cfg": SceneEntityCfg("robot", body_names=["FM_24_link"])
            }
        )  
        def __post_init__(self):
            super().__post_init__() # 计算维度等
            # Critic 观测不拼接，保持字典形式以便区分不同类型的输入
            self.concatenate_terms = True # 用于配置覆盖父类的设置
            self.enable_corruption = False # Critic 观测通常不加噪声
    
    @configclass
    class DebugCfg(ObsGroup):
        # 用于调试的观测项， 可以在测试时打印一些状态信息， 但不参与训练
        back_link_pose_error = ObsTerm(
            func = get_pose_error,
            params = {
                "asset_cfg": SceneEntityCfg("robot", body_names=["BM_01_link"])
            }
        )
    
    # 注册到总配置
    # 注意：SKRL能很好地处理这种字典输入
    camera: CameraCfg = CameraCfg()
    policy = PolicyCfg()
    # critic = CriticCfg()
    debug: DebugCfg = DebugCfg()
    