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
        # 基础速度 [Vx, Vy]
        self.cmd_vel = [0.0, 0.0] 
        # 最大线速度 m/s
        self.max_speed = 0.4 
        # 摇杆状态跟踪 (处理独立轴事件)
        self.stick_val = {
            "UP": 0.0, "DOWN": 0.0,
            "LEFT": 0.0, "RIGHT": 0.0
        }
        # 关节位置状态 (弧度)
        self.cmd_pos = {
            "pipe_dia_01": 0.8, # 后小臂 (Mid)
            "pipe_dia_02": 0.8, # 前小臂 (Mid)
            "up_arms_01":  0.0, # 后大臂 (Up)
            "up_arms_02":  0.0, # 前大臂 (Up)
            "bend_01":     0.0, # 后弯折
            "bend_02":     0.0, # 前弯折
        }
        self.cmd_pos_limits = {
            "pipe_dia_01": (-0.488692, 1.274090),
            "pipe_dia_02": (-0.488692, 1.274090),
            "up_arms_01":  (-0.6, 0.5),
            "up_arms_02":  (-0.6, 0.5),
            "bend_01":     (-0.488692, 1.047197533),
            "bend_02":     (-0.488692, 1.047197533),
        }
        # 步进角度 (弧度)
        self.step_small = math.radians(3.0)
        self.step_large = math.radians(5.0)

        # 重置标志
        self.needs_reset = False

        # 打印信息
        self.term_names = self.env.action_manager.active_terms
        print(f"[INFO] Action Terms: {self.term_names}")
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
                
                # 合成速度
                # Y轴 (前后): UP(+) / DOWN(-) -> 机器人 Y+ 是前进
                self.cmd_vel[1] = (self.stick_val["UP"] - self.stick_val["DOWN"]) * self.max_speed
                # X轴 (左右): LEFT(+) / RIGHT(-) -> 机器人 X+ 是右移
                self.cmd_vel[0] = (self.stick_val["RIGHT"] - self.stick_val["LEFT"]) * self.max_speed

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
                self.cmd_vel[1] = self.max_speed
            elif event.input == carb.input.KeyboardInput.S:
                self.cmd_vel[0] = 0.0  # 覆盖X轴，防止冲突
                self.cmd_vel[1] = -self.max_speed
            elif event.input == carb.input.KeyboardInput.A:
                self.cmd_vel[0] = -self.max_speed
                self.cmd_vel[1] = 0.0  # 覆盖Y轴，防止冲突
            elif event.input == carb.input.KeyboardInput.D:
                self.cmd_vel[0] = self.max_speed
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

    def run(self):
        """运行主循环"""
        obs, _ = self.env.reset()
        action_mgr = self.env.action_manager
        
        print("\n" + "="*50)
        print("Pipe Robot Advanced Control Demo")
        print(" Controls Reference:")
        print("  Base Move : L-Stick / WASD")
        print("  Rear Pipe : X/Y     / U/I (Dia)")
        print("  Front Pipe: A/B     / J/K (Dia)")
        print("  Rear Arm  : D-U/D   / Up/Down")
        print("  Front Arm : D-L/R   / Left/Right")
        print("  Rear Bend : LB/LT   / T/Y")
        print("  Front Bend: RB/RT   / G/H")

        if self.needs_reset:
            self.env.reset()
            self.needs_reset = False
            print("[INFO] Environment Reset Triggered.")

        count = 0
        while simulation_app.is_running():
            actions_list = []
            
            # # * --- Debug: Print cmd_vel periodically to ensure input is received ---
            # count += 1
            # if count % 50 == 0:
            #     # 打印当前所有要下发的指令，包括self.cmd_vel和self.cmd_pos
            #     print(  f"[CMD] Vel: ({self.cmd_vel[0]:.2f}, {self.cmd_vel[1]:.2f}) | "
            #             f"Pipe1: {self.cmd_pos['pipe_dia_01']:.2f} | "
            #             f"Pipe2: {self.cmd_pos['pipe_dia_02']:.2f} | "
            #             f"Arm1: {self.cmd_pos['up_arms_01']:.2f} | "
            #             f"Arm2: {self.cmd_pos['up_arms_02']:.2f} | "
            #             f"Bend1: {self.cmd_pos['bend_01']:.2f} | "
            #             f"Bend2: {self.cmd_pos['bend_02']:.2f}")
            
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
            obs, _, terminated, truncated, _ = self.env.step(full_action)
            
            count += 1
            if count % 100 == 0:
                obs_mgr = self.env.observation_manager
                
            if terminated.any() or truncated.any():
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


