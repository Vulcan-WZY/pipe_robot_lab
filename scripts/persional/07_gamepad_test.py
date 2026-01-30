# 2026.01.19: 手柄连接与输入测试脚本 (Final Clean Version)
import argparse
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Gamepad Input Test Script.")
# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""
必须在 app 启动后导入其他 Omniverse 模块
"""
import carb.input
import omni.appwindow

def on_input_event(event: carb.input.InputEvent, *args, **kwargs):
    """
    输入事件回调函数
    """
    # 1. 过滤掉非手柄事件
    # 注意: carb 绑定使用 deviceType (驼峰命名)
    if event.deviceType != carb.input.DeviceType.GAMEPAD:
        return True

    # 2. 获取内部的具体手柄事件数据
    # InputEvent.event -> GamepadEvent
    gp_event = event.event
    
    # 3. 提取数据
    try:
        val = gp_event.value
        input_id = gp_event.input
        input_name = str(input_id).replace("GamepadInput.", "")
        
        # 4. 简单的显示逻辑
        # 轴 (Axis) 通常值是连续浮点数，或者名字包含 STICK/TRIGGER
        is_axis = "STICK" in input_name or "TRIGGER" in input_name
        
        if is_axis:
            # 轴事件：应用死区过滤
            if abs(val) > 0.1:
                print(f"[Axis]   {input_name:<25} : {val:.4f}")
        else:
            # 按键事件：显示按下/释放状态
            # 有些按键也是模拟量的(如压感按键)，但大多是 0/1
            state = "PRESSED" if abs(val) > 0.5 else "RELEASED"
            print(f"[Button] {input_name:<25} : {state} ({val:.1f})")

    except Exception as e:
        print(f"[ERROR] Parse failed: {e}")

    return True

def main():
    print("\n" + "="*50)
    print("Gamepad Connection Test Tool")
    print("==================================================")
    print("[INFO] Initializing Input Interface...")
    
    # 获取输入接口
    input_interface = carb.input.acquire_input_interface()
    
    # 订阅输入事件
    # 使用位置参数以确保兼容性: (callback, event_types_mask, device, order)
    # 0xFFFFFFFF = 监听所有事件类型
    sub_id = input_interface.subscribe_to_input_events(
        on_input_event,
        0xFFFFFFFF, 
        None,       
        0           
    )
    
    print("[INFO] Listening for Gamepad inputs...")
    print("[INFO] Try moving joysticks or pressing buttons.")
    print("[INFO] (Axis values < 0.1 are ignored to prevent noise)")
    print("[INFO] Press Ctrl+C in terminal to exit.")
    print("="*50 + "\n")

    try:
        # 主循环：保持应用运行以接收事件
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected.")
    finally:
        # 清理订阅
        input_interface.unsubscribe_to_input_events(sub_id)
        simulation_app.close()
        print("[INFO] Exited.")

if __name__ == "__main__":
    main()
