import torch
import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp

# =============================================================================
# * 自定义动作项 (Custom Actions)
# =============================================================================

# 定义一些常量
MAIN_WHEEL_R    = 0.05 # 主轮半径 (米), 直径100mm
ASSIST_WHEEL_R  = 0.03 # 辅助轮半径 (米)，直径60mm

@configclass
class LinkedArmActionCfg(mdp.JointPositionActionCfg):
    """Configuration for LinkedArmAction."""
    class_type = None # Set below
    # 基础配置
    asset_name: str = "robot"
    # 控制mid_arm的关节 (同时也是 joint_names)
    joint_names: list[str] = [".*mid_arm_.*"] 
    # 跟随联动的tail_arm关节（会自动计算）
    tail_joint_names: list[str] = [".*tail_arm_.*"]
    def __post_init__(self):
        # 显式设置 implementation class
        self.class_type = LinkedArmAction
        super().__post_init__()
class LinkedArmAction(mdp.JointPositionAction):
    """自定义联动动作：控制mid_arm，自动计算tail_arm"""
    cfg: LinkedArmActionCfg
    def __init__(self, cfg: LinkedArmActionCfg, env: ManagerBasedRLEnv):
        # 初始化基类，只注册 mid_arm 关节 (由 cfg.joint_names 指定)
        super().__init__(cfg, env)
        # 获取tail_arm关节的索引
        asset = env.scene[cfg.asset_name]
        tail_joint_names = []
        for pattern in cfg.tail_joint_names:
            # find_joints 返回 (indices, names)
            ids, _ = asset.find_joints(pattern)
            tail_joint_names.extend(ids)
        # 存储tail_arm关节的索引
        self.tail_joint_idxs = torch.tensor(tail_joint_names, device=env.device, dtype=torch.int32)
        # mid_arm索引存储在 self._joint_ids (基类处理)
        print(f"[INFO] LinkedArmAction: {len(self._joint_ids)} mid_arms -> {len(self.tail_joint_idxs)} tail_arms")

        # 注意: 如果 _joint_ids 是 tensor，需先转为 list
        mid_ids_list = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        tail_ids_list = self.tail_joint_idxs.tolist() if isinstance(self.tail_joint_idxs, torch.Tensor) else self.tail_joint_idxs
        
        mid_names = [asset.joint_names[i] for i in mid_ids_list]
        tail_names = [asset.joint_names[i] for i in tail_ids_list]
        
        print(f"[INFO] LinkedArmAction Pairing Check ({len(mid_names)} pairs):")
        # 打印配对情况，便于用户检查是否错位（如 mid_arm_01 对应到了 tail_arm_02）
        for i, (m, t) in enumerate(zip(mid_names, tail_names)):
            print(f"  Pair {i:02d}: {m:<30} --> {t}")
        
        if len(mid_names) != len(tail_names):
                print(f"[WARNING] LinkedArmAction: Mismatch in number of joints! Mid: {len(mid_names)}, Tail: {len(tail_names)}")
    def apply_actions(self):
        # 1. 应用 mid_arm (基类逻辑)
        # process_actions 已经在 step 前被调用，self.processed_actions 包含 mid_arm 的目标位置
        super().apply_actions()
        mid_targets = self.processed_actions
        # 计算并配置 tail targets
        current_mid = self._asset.data.joint_pos[:, self._joint_ids]
        tail_targets = self._compute_tail_position(current_mid)
        self._asset.set_joint_position_target(tail_targets, joint_ids=self.tail_joint_idxs)
    def _compute_tail_position(self, mid_positions: torch.Tensor) -> torch.Tensor:
        """根据mid_arm位置计算tail_arm位置的自定义函数"""
        AB = 26.88
        BC = 151.5
        CD = 37
        DA = 162
        # 四连杆机构，mid对应角ABC， tail对应(pi - angleBCD)
        # 角ABC的实际值等于 mid_arm + 57.63°， tail为0时， (pi - angleBCD) = 47.87°
        # 计算 mid_arm 对应的 angleABC
        angle_ABC = mid_positions + math.radians(57.63)
        # 使用余弦定理计算 对角线 AC 的长度
        AC = torch.sqrt(AB**2 + BC**2 - 2*AB*BC*torch.cos(angle_ABC))
        # 进一步使用余弦定理分别计算 angleBCA和angele ACD
        angle_BCA = torch.acos((BC**2 + AC**2 - AB**2) / (2 * BC * AC))
        angle_ACD = torch.acos((AC**2 + CD**2 - DA**2) / (2 * AC * CD))
        # 计算 angleBCD
        angle_BCD = angle_BCA + angle_ACD
        # 计算 tail_arm 位置 (tail_positions)
        tail_positions = math.pi - angle_BCD
        # 使用余弦定理计算 angleBCD
        # 转换为相对于默认位置的偏移量
        return tail_positions - math.radians(47.87)

@configclass
class SteerWheelActionCfg(mdp.JointVelocityActionCfg):
    # 配置一组舵轮联合控制
    asset_name: str = "robot"
    # 轮电机
    joint_names: list[str] = [".*wheel_.*"]
    # 与轮电机配套对应的舵电机
    steer_joint_names: list[str] = [".*steer_.*"]
    scale: float = 1.0
    def __post_init__(self):
        self.class_type = SteerWheelAction
        super().__post_init__()
class SteerWheelAction(mdp.JointVelocityAction):
    cfg: SteerWheelActionCfg
    def __init__(self , cfg: SteerWheelActionCfg, env: ManagerBasedRLEnv):
        # 1. 初始化基类
        super().__init__(cfg, env)
        # 2. 获取配套的舵关节索引
        asset = env.scene[cfg.asset_name]
        steer_joint_names = []
        for pattern in cfg.steer_joint_names:
            ids, _ = asset.find_joints(pattern)
            steer_joint_names.extend(ids)
        # 3. 存储索引
        self.steer_joint_idxs = torch.tensor(steer_joint_names, device=env.device, dtype=torch.int32)
        
        # --- Debug Info ---
        wheel_ids = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        steer_ids = self.steer_joint_idxs.tolist() if isinstance(self.steer_joint_idxs, torch.Tensor) else self.steer_joint_idxs
        
        print(f"[DEBUG] SteerWheelAction Init: WheelPattern={cfg.joint_names}, SteerPattern={cfg.steer_joint_names}")
        print(f"        -> Found Wheels: {wheel_ids}, Steers: {steer_ids}")
        
        if len(wheel_ids) == 0:
            print(f"[ERROR] No WHEEL joints found matching {cfg.joint_names}")
        if len(steer_ids) == 0:
            print(f"[ERROR] No STEER joints found matching {cfg.steer_joint_names}")

        # 确保轮子数量和舵数量一致
        if len(self._joint_ids) != len(self.steer_joint_idxs) and len(wheel_ids) > 0:
            raise ValueError(f"Number of wheels ({len(self._joint_ids)}) and steers ({len(self.steer_joint_idxs)}) must match!")
        # 4. 打印配对信息
        # 修正：当只有一个索引且为 list 类型时 (基类 behavior 可能不同)，不要使用 .tolist()
        # 基类 mdp.JointVelocityAction 的 _joint_ids 可能是 torch.Tensor 也可能是 list
        # 安全地将其转换为 list
        wheel_ids = self._joint_ids.tolist() if isinstance(self._joint_ids, torch.Tensor) else self._joint_ids
        steer_ids = self.steer_joint_idxs.tolist() if isinstance(self.steer_joint_idxs, torch.Tensor) else self.steer_joint_idxs
        
        wheel_names = [asset.joint_names[i] for i in wheel_ids]
        steer_names = [asset.joint_names[i] for i in steer_ids]
        
        if len(wheel_names) > 0 and len(steer_names) > 0:
            print(f"[INFO] SteerWheelAction Unit: {wheel_names[0]} <--> {steer_names[0]}")

    @property
    def action_dim(self):
        return 2
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._processed_actions = actions * self.cfg.scale
        return self._processed_actions
    def apply_actions(self):
        # Input: [Vx, Vy] -> (Num_Envs, 2)
        cmds = self.processed_actions
        vx = cmds[:, 0]
        vy = cmds[:, 1]
        
        # --- Configs & Thresholds ---
        # 最小输入阈值 (Input Deadzone)
        CMD_THRESHOLD = 0.01
        # 等待舵轮转动到位的阈值 (Wait-for-Align), 0.2 rad ≈ 11.5度
        WAIT_ANGLE_THRESHOLD = 0.2 
        # 舵机机械限位 (Joint Limit), URDF defined [-1.8, 1.8]
        JOINT_LIMIT = 1.8
        
        # 1. 基础 Swerve 解算 (Basic Kinematics)
        # 目标线速度 V
        cmd_speed = torch.hypot(vx, vy).unsqueeze(1) # (Num_Envs, 1)
        # 理想目标角度 (相对于 +Y 轴)
        # atan2(y, x) 是相对于 +X 的角度. 当 vx=0, vy=1 (向前)时, atan2=pi/2.
        # 此时我们需要 0 度. 所以 -pi/2.
        ideal_angle = torch.atan2(vy, vx).unsqueeze(1) - 0.5 * math.pi
        # Normalize to [-pi, pi]
        ideal_angle = (ideal_angle + math.pi) % (2 * math.pi) - math.pi
        
        # 2. 机械限位与反转逻辑 (Limit & Reverse Logic)
        # 由于舵机只能转 +/- 1.8 rad (约103度), 也就是只能覆盖前方的扇区.
        # 如果目标在后方, 我们必须"反向驱动": 舵机转到反方向, 电机反转.
        
        # 方案 A: 正向驱动 (Forward) -> 目标为 ideal_angle
        # 方案 B: 反向驱动 (Reverse) -> 目标为 ideal_angle + pi
        
        flip_angle = (ideal_angle + math.pi + math.pi) % (2 * math.pi) - math.pi
        
        # 检查物理可行性 (Is physically reachable?)
        valid_A = torch.abs(ideal_angle) <= JOINT_LIMIT
        valid_B = torch.abs(flip_angle) <= JOINT_LIMIT
        
        # 3. 现在的状态 (Current State)
        current_pos = self._asset.data.joint_pos[:, self.steer_joint_idxs]
        
        # 计算移动代价 (Distance to target)
        dist_A = torch.abs(ideal_angle - current_pos)
        dist_B = torch.abs(flip_angle - current_pos)
        
        # 选择逻辑:
        # 1. 如果 B 可行, 且 (A 不可行 或 B 更近), 则选 B
        # 2. 否则选 A (假设 A 可行, 或者都不可行时随便选一个)
        # 注: [-1.8, 1.8] 范围 > 180度, 理论上 A/B 必有一个可行.
        use_B = valid_B & (~valid_A | (dist_B < dist_A))
        
        target_angle = torch.where(use_B, flip_angle, ideal_angle)
        target_speed = torch.where(use_B, -cmd_speed, cmd_speed)
        
        # 4. 等待对齐逻辑 (Wait-for-Align)
        # 如果当前角度误差过大, 此时强行转动轮子会导致打滑/跳动/摩擦对抗
        # 策略: 误差大时, 轮速置零
        err = torch.abs(target_angle - current_pos)
        final_speed = torch.where(err > WAIT_ANGLE_THRESHOLD, torch.zeros_like(target_speed), target_speed)
        
        # 5. 静止保持逻辑 (Standstill Handling)
        # 如果输入非常小(摇杆回中), 不应该乱转舵机, 应该保持当前位置或锁定
        active_mask = cmd_speed > CMD_THRESHOLD
        
        # 如果 active: 使用计算出的 target_angle
        # 如果 inactive: 使用 current_pos (锁定当前位置) 防止回中时归零
        final_angle_cmd = torch.where(active_mask, target_angle, current_pos)
        
        # 6. 下发指令
        self._asset.set_joint_velocity_target(final_speed, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(final_angle_cmd, joint_ids=self.steer_joint_idxs)

# =============================================================================
# * MDP Action 注册
# =============================================================================

@configclass
class ActionsCfg:
    # -------------------------------------------------------------------------
    # 舵轮阵列 (Swerve Drive Modules)
    # 请根据实际关节名称，复制 6 份并修改正则
    # -------------------------------------------------------------------------
    # 使用 SteerWheelActionCfg，每个实例仅控制一对 [Wheel, Steer]
    steer_wheel_01 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_01"],        # 左后辅助轮
        steer_joint_names=[".*assist_steer_01"],  # 左后辅助轮舵向
        # 通过scale将线速度转换为角速度： wheel_ang_vel = line_vel / wheel_radius
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_02 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_02"],        # 右后辅助轮
        steer_joint_names=[".*assist_steer_02"],  # 右后辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_03 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*main_wheel_01"],          # 中后主动轮
        steer_joint_names=[".*main_steer_01"],    # 中后主动轮舵向
        scale=1.0 / MAIN_WHEEL_R,
    )
    steer_wheel_04 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_03"],          # 左前辅助轮
        steer_joint_names=[".*assist_steer_03"],    # 左前辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_05 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*assist_wheel_04"],          # 右前辅助轮
        steer_joint_names=[".*assist_steer_04"],    # 右前辅助轮
        scale=1.0 / ASSIST_WHEEL_R,
    )
    steer_wheel_06 = SteerWheelActionCfg(
        asset_name="robot",
        joint_names=[".*main_wheel_02"],          # 中前主动轮
        steer_joint_names=[".*main_steer_02"],    # 中前主动轮
        scale=1.0 / MAIN_WHEEL_R,
    )
    # -------------------------------------------------------------------------
    # ! 机械臂与弯折 (Manipulators)
    # -------------------------------------------------------------------------
    # * 联动臂 实际控制关节数量: 2 * 4 = 8
    # 后侧两组联动臂
    pipe_dia_01 = LinkedArmActionCfg(
        asset_name="robot",
        joint_names=["mid_arm_01", "mid_arm_02"],
        tail_joint_names=["tail_arm_01", "tail_arm_02"],
        scale=1.0,
        use_default_offset=True,
    )
    # 前侧两组联动臂
    pipe_dia_02 = LinkedArmActionCfg(
        asset_name="robot",
        joint_names=["mid_arm_03", "mid_arm_04"],
        tail_joint_names=["tail_arm_03", "tail_arm_04"],
        scale=1.0,
        use_default_offset=True,
    )
    # * 两组大臂控制 实际控制关节数量: 2 * 2 = 4
    up_arms_01 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*up_arm_01", ".*up_arm_02"],
        scale=1.0,
        use_default_offset=True, 
    )
    up_arms_02 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*up_arm_03", ".*up_arm_04"],
        scale=1.0,
        use_default_offset=True, 
    )
    # * 机身弯折控制
    bend_01 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["bend_01"],
        scale=1.0,
        use_default_offset=True,
    )
    bend_02 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["bend_02"],
        scale=1.0,
        use_default_offset=True,
    )
    # =============================================================================

