# 2026.01.18: 经过初步测试后形成的完整Demo运行文件，期望配置管道检测机器人可按照按键控制夹持在管道表面并沿管道前进
import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Pipe Robot Control Demo Script.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import carb.input
import omni.appwindow # 新增导入
import math
import cv2  # 用于保存图片
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
# from pipe_robot_env import PipeRobotEnvCfg
from pipe_robot_lab.tasks.manager_based.pipe_robot_lab.pipe_robot_lab_env_cfg import PipeRobotLabEnvCfg


class PiprRobotDemo:
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        
        # 1. 订阅键盘事件
        self.sub_kwd = self.input.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )
        # 2. 订阅通用事件 (用于手柄)
        self.sub_pad = self.input.subscribe_to_input_events(
            self._on_gamepad_event,
            0xFFFFFFFF, 
            None,       
            0           
        )

        # --- 控制指令状态 (Command States) ---
        # 基础速度 [Vx, Vy], 归一化输入 [-1, 1], 用于模拟 RL 网络的第一、第二位输出
        self.cmd_vel = [0.0, 0.0] 
        
        # 摇杆状态跟踪 (处理独立轴事件)
        self.stick_val = {
            "UP": 0.0, "DOWN": 0.0,
            "LEFT": 0.0, "RIGHT": 0.0
        }
        # 关节位置状态, 全部采用归一化 [-1, 1], 用于模拟 RL 网络动作输出
        self.cmd_pos = {
            "pipe_dia_01": 0.0, # 后小臂 (Mid) 0对应默认中位(offset)
            "pipe_dia_02": 0.0, # 前小臂 (Mid)
            "up_arms_01":  0.0, # 后大臂 (Up)
            "up_arms_02":  0.0, # 前大臂 (Up)
            "bend_01":     0.0, # 后弯折
            "bend_02":     0.0, # 前弯折
        }
        self.cmd_pos_limits = {
            "pipe_dia_01": (-1.0, 1.0),
            "pipe_dia_02": (-1.0, 1.0),
            "up_arms_01":  (-1.0, 1.0),
            "up_arms_02":  (-1.0, 1.0),
            "bend_01":     (-1.0, 1.0),
            "bend_02":     (-1.0, 1.0),
        }
        # 增加步进响应映射(每次按键操作改变归一化数值的量)
        self.step_small = 0.05
        self.step_large = 0.1

        # 重置标志
        self.needs_reset = False

        # 打印信息
        self.term_names = self.env.action_manager.active_terms
        print(f"[INFO] Action Terms: {self.term_names}")
        self.rew_term_names = []
        if hasattr(self.env, "reward_manager") and hasattr(self.env.reward_manager, "active_terms"):
            self.rew_term_names = list(self.env.reward_manager.active_terms)
            print(f"[INFO] Reward Terms: {self.rew_term_names}")
        self._prev_episode_sums = None
        print("[INFO] Gamepad & Keyboard Control Ready.")

    def _update_cmd_pos(self, key, delta):
        """辅助函数：更新关节位置并进行限位限制"""
        if key in self.cmd_pos and key in self.cmd_pos_limits:
            low, high = self.cmd_pos_limits[key]
            self.cmd_pos[key] = max(low, min(high, self.cmd_pos[key] + delta))

    def _on_gamepad_event(self, event: carb.input.InputEvent, *args, **kwargs):
        """手柄事件处理"""
        if event.deviceType != carb.input.DeviceType.GAMEPAD:
            return True
        gp_event = event.event
        try:
            val = gp_event.value
            input_id = gp_event.input
            # 获取简洁的名称 (e.g. "LEFT_STICK_UP", "A", "DPAD_UP")
            name = str(input_id).replace("GamepadInput.", "")
            # -------------------------------------------------
            # * 1. 摇杆控制底盘移动 (Left Stick)
            # -------------------------------------------------
            # 更新轴状态
            if name in ["LEFT_STICK_UP", "LEFT_STICK_DOWN", "LEFT_STICK_LEFT", "LEFT_STICK_RIGHT"]:
                # 简单的死区过滤
                stick_v = val if abs(val) > 0.05 else 0.0
                
                if name == "LEFT_STICK_UP":    self.stick_val["UP"] = stick_v
                if name == "LEFT_STICK_DOWN":  self.stick_val["DOWN"] = stick_v
                if name == "LEFT_STICK_LEFT":  self.stick_val["LEFT"] = stick_v
                if name == "LEFT_STICK_RIGHT": self.stick_val["RIGHT"] = stick_v
                
                # 合成速度, 归一化输入范围 [-1, 1]
                # Y轴 (前后): UP(+) / DOWN(-) -> 机器人 Y+ 是前进
                self.cmd_vel[1] = (self.stick_val["UP"] - self.stick_val["DOWN"])
                # X轴 (左右): LEFT(+) / RIGHT(-) -> 机器人 X+ 是右移
                self.cmd_vel[0] = (self.stick_val["RIGHT"] - self.stick_val["LEFT"])

            # -------------------------------------------------
            # * 2. 按钮触发离散动作
            # -------------------------------------------------
            is_pressed = val > 0.5
            if not is_pressed:
                return True
            # --- 2. 后侧小臂 (XY / UI) -> 对应手柄 X/Y ---
            # X: 缩小
            if name == "X":
                self._update_cmd_pos("pipe_dia_01", -self.step_large)
            # Y: 变大
            elif name == "Y":
                self._update_cmd_pos("pipe_dia_01", self.step_large)
            # --- 3. 前侧小臂 (AB / JK) -> 对应手柄 A/B ---
            # A: 缩小
            elif name == "A":
                self._update_cmd_pos("pipe_dia_02", -self.step_large)
            # B: 变大
            elif name == "B":
                self._update_cmd_pos("pipe_dia_02", self.step_large)
            # --- 4. 后侧大臂 (DPAD U/D) ---
            elif name == "DPAD_UP":
                self._update_cmd_pos("up_arms_01", self.step_small)
            elif name == "DPAD_DOWN":
                self._update_cmd_pos("up_arms_01", -self.step_small)
            # --- 5. 前侧大臂 (DPAD L/R) ---
            elif name == "DPAD_RIGHT":
                self._update_cmd_pos("up_arms_02", self.step_small)
            elif name == "DPAD_LEFT":
                self._update_cmd_pos("up_arms_02", -self.step_small)
            # --- 6. 后侧弯折 (LB/LT) ---
            elif name == "LEFT_SHOULDER": # LB
                self._update_cmd_pos("bend_01", -self.step_small)
            elif name == "LEFT_TRIGGER":  # LT
                self._update_cmd_pos("bend_01", self.step_small)
            # --- 7. 前侧弯折 (RB/RT) ---
            elif name == "RIGHT_SHOULDER": # RB
                self._update_cmd_pos("bend_02", -self.step_small)
            elif name == "RIGHT_TRIGGER":  # RT
                self._update_cmd_pos("bend_02", self.step_small)
        except Exception:
            pass
        return True

    def _on_keyboard_event(self, event, *args, **kwargs):
        """键盘事件处理"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 1. 移动 (WASD Full Speed)
            if event.input == carb.input.KeyboardInput.W:
                self.cmd_vel[0] = 0.0  # 覆盖X轴，防止冲突
                self.cmd_vel[1] = 1.0  # 归一化输入范围 [-1, 1]
            elif event.input == carb.input.KeyboardInput.S:
                self.cmd_vel[0] = 0.0  # 覆盖X轴，防止冲突
                self.cmd_vel[1] = -1.0 # 归一化输入范围 [-1, 1]
            elif event.input == carb.input.KeyboardInput.A:
                self.cmd_vel[0] = -1.0 # 归一化输入范围 [-1, 1]
                self.cmd_vel[1] = 0.0  # 覆盖Y轴，防止冲突
            elif event.input == carb.input.KeyboardInput.D:
                self.cmd_vel[0] = 1.0  # 归一化输入范围 [-1, 1]
                self.cmd_vel[1] = 0.0  # 覆盖Y轴，防止冲突

            # 2. 后侧小臂 (U/I)
            elif event.input == carb.input.KeyboardInput.U:
                self._update_cmd_pos("pipe_dia_01", -self.step_large)
            elif event.input == carb.input.KeyboardInput.I:
                self._update_cmd_pos("pipe_dia_01", self.step_large)
            
            # 3. 前侧小臂 (J/K)
            elif event.input == carb.input.KeyboardInput.J:
                self._update_cmd_pos("pipe_dia_02", -self.step_large)
            elif event.input == carb.input.KeyboardInput.K:
                self._update_cmd_pos("pipe_dia_02", self.step_large)

            # 4. 后侧大臂 (UP/DOWN)
            elif event.input == carb.input.KeyboardInput.UP:
                self._update_cmd_pos("up_arms_01", self.step_small)
            elif event.input == carb.input.KeyboardInput.DOWN:
                self._update_cmd_pos("up_arms_01", -self.step_small)

            # 5. 前侧大臂 (LEFT/RIGHT)
            elif event.input == carb.input.KeyboardInput.RIGHT:
                self._update_cmd_pos("up_arms_02", self.step_small)
            elif event.input == carb.input.KeyboardInput.LEFT:
                self._update_cmd_pos("up_arms_02", -self.step_small)

            # 6. 后侧弯折 (T/Y)
            elif event.input == carb.input.KeyboardInput.T:
                self._update_cmd_pos("bend_01", -self.step_small)
            elif event.input == carb.input.KeyboardInput.Y:
                self._update_cmd_pos("bend_01", self.step_small)
            
            # 7. 前侧弯折 (G/H)
            elif event.input == carb.input.KeyboardInput.G:
                self._update_cmd_pos("bend_02", -self.step_small)
            elif event.input == carb.input.KeyboardInput.H:
                self._update_cmd_pos("bend_02", self.step_small)

            # 8. 重置 (R)
            elif event.input == carb.input.KeyboardInput.R:
                self.needs_reset = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # 停止移动
            if event.input in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]:
                self.cmd_vel[0] = 0.0
                self.cmd_vel[1] = 0.0
            if event.input in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]:
                self.cmd_vel[0] = 0.0
                self.cmd_vel[1] = 0.0
        return True

    def _get_obs_terms_meta(self, obs_mgr, group_name: str):
        """获取指定 Observation 组的 term 名称与维度元数据。"""
        # term names
        if hasattr(obs_mgr, "group_obs_term_names"):
            term_names_list = obs_mgr.group_obs_term_names.get(group_name, [])
        elif hasattr(obs_mgr, "_group_obs_term_names"):
            term_names_list = obs_mgr._group_obs_term_names.get(group_name, [])
        else:
            term_names_list = []

        # term dims
        if hasattr(obs_mgr, "_group_obs_term_dim"):
            term_dims_list = obs_mgr._group_obs_term_dim.get(group_name, [])
        elif hasattr(obs_mgr, "group_obs_term_dim"):
            term_dims_list = obs_mgr.group_obs_term_dim.get(group_name, [])
        else:
            if hasattr(obs_mgr, "group_obs_dim"):
                term_dims_list = obs_mgr.group_obs_dim.get(group_name, [])
            else:
                term_dims_list = []

        if not term_names_list and term_dims_list:
            print("    [Info] Names not found, printing by index chunks based on dims.")
            term_names_list = [f"Term_{i}" for i in range(len(term_dims_list))]

        if len(term_names_list) != len(term_dims_list):
            print(f"    [Warning] Names/Dims mismatch: {len(term_names_list)} vs {len(term_dims_list)}")

        return term_names_list, term_dims_list

    def _print_observation_group(self, obs: dict, group_name: str, env_idx: int = 0):
        """打印指定 Observation 组的 shape 与按 term 切片后的数值。"""
        group_obs = obs.get(group_name)
        group_name_upper = group_name.capitalize()

        if isinstance(group_obs, torch.Tensor):
            print(f"  > [{group_name_upper}] Tensor Shape: {tuple(group_obs.shape)}")
            obs_mgr = self.env.observation_manager
            try:
                term_names_list, term_dims_list = self._get_obs_terms_meta(obs_mgr, group_name)

                current_idx = 0
                for name, dims in zip(term_names_list, term_dims_list):
                    length = int(np.prod(dims))
                    end_idx = current_idx + length

                    if end_idx <= group_obs.shape[1]:
                        val = group_obs[env_idx, current_idx:end_idx].cpu().numpy()
                        print(f"    - {name:<18} [{length:2d}]: {np.array2string(val, precision=3, suppress_small=True)}")
                    else:
                        print(f"    - {name:<18}: [Index Error] {current_idx}:{end_idx} vs {group_obs.shape[1]}")

                    current_idx = end_idx
            except Exception as e:
                print(f"    [Smart Print Failed]: {e}")
                print(f"    Available attributes: {[k for k in dir(obs_mgr) if 'group' in k or 'term' in k or 'dim' in k]}")
        elif group_obs is None:
            print(f"  > [{group_name_upper}] Key not found in obs.")
        else:
            print(f"  > [{group_name_upper}] Unexpected type: {type(group_obs)}")

    def _get_reward_terms_meta(self):
        """获取 Reward term 名称与权重，风格对齐 observation term 元数据函数。"""
        term_names = []
        term_weights = {}

        rew_mgr = getattr(self.env, "reward_manager", None)
        if rew_mgr is None:
            return term_names, term_weights

        if hasattr(rew_mgr, "active_terms"):
            term_names = list(rew_mgr.active_terms)
        elif hasattr(rew_mgr, "_term_names"):
            term_names = list(rew_mgr._term_names)

        # 读取配置权重（如果可用）
        term_cfgs = getattr(rew_mgr, "_term_cfgs", None)
        if term_cfgs is not None:
            for i, name in enumerate(term_names):
                if i < len(term_cfgs) and hasattr(term_cfgs[i], "weight"):
                    term_weights[name] = float(term_cfgs[i].weight)

        return term_names, term_weights

    def _extract_reward_term_values(self, extras: dict, env_idx: int, term_names: list[str]):
        """从 extras 中提取每个 reward term 的值（尽量不依赖固定结构）。"""
        values = {}
        if not isinstance(extras, dict):
            return values

        def _get_scalar(v):
            if isinstance(v, torch.Tensor):
                flat = v.view(-1)
                if flat.numel() == 0:
                    return None
                idx = env_idx if flat.numel() > env_idx else 0
                return float(flat[idx].item())
            if isinstance(v, (int, float)):
                return float(v)
            return None

        # 深度优先扫描 dict，支持 key 为 term_name 或以 "/term_name" 结尾
        stack = [extras]
        while stack:
            cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            for k, v in cur.items():
                if isinstance(v, dict):
                    stack.append(v)
                    continue
                for term_name in term_names:
                    if k == term_name or str(k).endswith(f"/{term_name}"):
                        scalar = _get_scalar(v)
                        if scalar is not None:
                            values[term_name] = scalar

        return values

    def _extract_reward_episode_sums(self, env_idx: int, term_names: list[str]):
        """从 reward_manager 的累计缓存提取每个 term 的累计奖励（回退路径）。"""
        values = {}
        rew_mgr = getattr(self.env, "reward_manager", None)
        if rew_mgr is None:
            return values

        episode_sums = getattr(rew_mgr, "_episode_sums", None)
        if not isinstance(episode_sums, dict):
            return values

        for term_name in term_names:
            if term_name in episode_sums:
                v = episode_sums[term_name]
                if isinstance(v, torch.Tensor):
                    flat = v.view(-1)
                    if flat.numel() > 0:
                        idx = env_idx if flat.numel() > env_idx else 0
                        values[term_name] = float(flat[idx].item())

        return values

    def _extract_reward_step_values_from_episode_delta(self, env_idx: int, term_names: list[str]):
        """通过 episode_sums 差分近似当前步的 reward term 值。"""
        rew_mgr = getattr(self.env, "reward_manager", None)
        if rew_mgr is None:
            return {}
        episode_sums = getattr(rew_mgr, "_episode_sums", None)
        if not isinstance(episode_sums, dict):
            return {}

        current = {}
        for term_name in term_names:
            v = episode_sums.get(term_name, None)
            if isinstance(v, torch.Tensor):
                flat = v.view(-1)
                if flat.numel() > 0:
                    idx = env_idx if flat.numel() > env_idx else 0
                    current[term_name] = float(flat[idx].item())

        if not current:
            return {}

        if self._prev_episode_sums is None:
            self._prev_episode_sums = current.copy()
            # 首帧无法差分，返回 0
            return {k: 0.0 for k in current}

        delta = {}
        for term_name, cur_v in current.items():
            prev_v = self._prev_episode_sums.get(term_name, cur_v)
            # 逐步差分应允许正负号；负值常见于惩罚项
            delta[term_name] = cur_v - prev_v

        self._prev_episode_sums = current.copy()
        return delta

    def _print_reward_group(self, step_reward, extras: dict, env_idx: int = 0):
        """按 term 范式打印每步总奖励与各项奖励。"""
        if isinstance(step_reward, torch.Tensor):
            reward_flat = step_reward.view(-1)
            if reward_flat.numel() > env_idx:
                print(f"  > [Reward] total_step_reward: {reward_flat[env_idx].item():.4f}")
            else:
                print(f"  > [Reward] total_step_reward tensor shape: {tuple(step_reward.shape)}")
        else:
            print(f"  > [Reward] total_step_reward: {step_reward}")

        term_names, term_weights = self._get_reward_terms_meta()
        if not term_names:
            print("  > [Reward Terms] reward_manager term metadata not available")
            if isinstance(extras, dict):
                print(f"  > [Reward Terms] extras keys: {list(extras.keys())}")
            return

        term_values = self._extract_reward_term_values(extras, env_idx, term_names)
        source = "extras(step)"

        # extras 往往是 episode 口径，这里优先回退到 episode_sums 差分，得到更接近实时的步级值
        all_near_zero = term_values and all(abs(v) < 1e-10 for v in term_values.values())
        if (not term_values) or all_near_zero:
            delta_values = self._extract_reward_step_values_from_episode_delta(env_idx, term_names)
            if delta_values:
                term_values = delta_values
                source = "reward_manager._episode_sums_delta(step-like)"
            else:
                term_values = self._extract_reward_episode_sums(env_idx, term_names)
                source = "reward_manager._episode_sums(cumulative)"

        print(f"  > [Reward Terms] ({source}):")
        for term_name in term_names:
            w = term_weights.get(term_name, None)
            w_str = f"w={w:.3f}" if w is not None else "w=?"
            if term_name in term_values:
                print(f"    - {term_name:<24} ({w_str}): {term_values[term_name]:.4f}")
            else:
                print(f"    - {term_name:<24} ({w_str}): N/A")

        if source.startswith("extras") and isinstance(extras, dict) and not term_values:
            print(f"  > [Reward Terms] extras keys: {list(extras.keys())}")

    def run(self):
        """运行主循环"""
        obs, _ = self.env.reset()
        action_mgr = self.env.action_manager
        
        # print("\n" + "="*50)
        # print("Pipe Robot Advanced Control Demo")
        # print(" Controls Reference:")
        # print("  Base Move : L-Stick / WASD")
        # print("  Rear Pipe : X/Y     / U/I (Dia)")
        # print("  Front Pipe: A/B     / J/K (Dia)")
        # print("  Rear Arm  : D-U/D   / Up/Down")
        # print("  Front Arm : D-L/R   / Left/Right")
        # print("  Rear Bend : LB/LT   / T/Y")
        # print("  Front Bend: RB/RT   / G/H")

        if self.needs_reset:
            self.env.reset()
            self.needs_reset = False
            print("[INFO] Environment Reset Triggered.")

        count = 0
        print_interval = 1000
        import time
        start_time = time.time()
        while simulation_app.is_running():
            actions_list = []
            
            # # * --- Debug: Print cmd_vel periodically to ensure input is received ---
            count += 1
            if count % print_interval == 0:
                current_time = time.time()
                avg_time = (current_time - start_time) / print_interval
                fps = 1.0 / avg_time if avg_time > 0 else 0.0
                print(f"\n[Timing] {print_interval} Frames Avg Time: {avg_time*1000:.2f} ms | FPS: {fps:.1f}")
                start_time = current_time
                
                # 打印当前所有要下发的指令，包括self.cmd_vel和self.cmd_pos
                # print(  f"[CMD] Vel: ({self.cmd_vel[0]:.2f}, {self.cmd_vel[1]:.2f}) | "
                #         f"Pipe1: {self.cmd_pos['pipe_dia_01']:.2f} | "
                #         f"Pipe2: {self.cmd_pos['pipe_dia_02']:.2f} | "
                #         f"Arm1: {self.cmd_pos['up_arms_01']:.2f} | "
                #         f"Arm2: {self.cmd_pos['up_arms_02']:.2f} | "
                #         f"Bend1: {self.cmd_pos['bend_01']:.2f} | "
                #         f"Bend2: {self.cmd_pos['bend_02']:.2f}")
            
            for term_name in self.term_names:
                term = action_mgr.get_term(term_name)
                dim = term.action_dim 
                term_action = torch.zeros((self.env.num_envs, dim), device=self.env.device)
                # --- 映射逻辑 ---
                if term_name.startswith("steer_wheel"):
                    # 舵轮模块：输入 [Vx, Vy]
                    term_action[:, 0] = self.cmd_vel[0]
                    term_action[:, 1] = self.cmd_vel[1]               
                elif term_name in self.cmd_pos:
                    # 位置控制模块：直接查字典
                    term_action[:] = self.cmd_pos[term_name]
                else:
                    # 其他未知项 (如 coupled_arms 被拆了?)
                    # 注意: 我们代码里注册的是 pipe_dia_01, 而不是 coupled_arms
                    pass
                actions_list.append(term_action)
            # 执行
            full_action = torch.cat(actions_list, dim=1)
            obs, step_reward, terminated, truncated, extras = self.env.step(full_action)

            
            # count += 1
            if count % print_interval == 0:
                print(f"[Frame {count}] Observation Debug:")

                # 打印奖励（总奖励 + 奖励分项）
                self._print_reward_group(step_reward, extras)

                # 只需要传入组名即可自动切片并打印
                # self._print_observation_group(obs, "policy")
                # self._print_observation_group(obs, "critic")
                self._print_observation_group(obs, "debug")

                # ---------------------------------------------------------
                # ! 3. 检查 Camera 组 (Dict)
                # ---------------------------------------------------------
                # camera_obs = obs.get("camera")
                # if isinstance(camera_obs, dict):
                #     print(f"  > [Camera] Group:")
                #     for key, val in camera_obs.items():
                #         if isinstance(val, torch.Tensor):
                #             print(f"    - {key}: {tuple(val.shape)}")
            
            if terminated.any() or truncated.any():
                self._prev_episode_sums = None
                obs, _ = self.env.reset()



def main():
    # 1. 实例化配置
    env_cfg = PipeRobotLabEnvCfg()
    # 2. 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # 3. 启动交互 Demo
    demo = PiprRobotDemo(env)
    demo.run()
    
    # 4. 退出
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()


