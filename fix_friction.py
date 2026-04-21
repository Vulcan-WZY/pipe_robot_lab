# ===========
# Date: 2026-04-21 16:43
# Author: Vulcan
# LastEditTime: 2026-04-21 16:50
# Description: 
# ==========
import os
import argparse
from isaaclab.app import AppLauncher

# 在导入 pxr 之前，必须先启动 AppLauncher 来注入 Omniverse (USD) 的路径
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Sdf

def fix_robot_usd(usd_path):
    print(f"Loading USD from: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("Failed to open stage!")
        return
        
    root_prim_path = "/pipe_robot" # 根据你的USD根节点名称可能需要调整
    
    # 1. 尝试删除旧的、会导致克隆器报错的 Material Prim
    material_path = f"{root_prim_path}/WheelMaterial"
    old_mat_prim = stage.GetPrimAtPath(material_path)
    if old_mat_prim.IsValid():
        stage.RemovePrim(material_path)
        print(f"Removed old Material prim at {material_path}")
    else:
        print(f"No old Material prim found at {material_path}, skipping deletion.")
        
    # 2. 精准定位 6 个轮子的碰撞网格节点 (collisions)
    wheel_links = [
        "BM_08_link",
        "FM_31_link",
        "BL_22_link",
        "BR_23_link",
        "FL_44_link",
        "FL_45_link"
    ]
    
    for link_name in wheel_links:
        # 你的截图显示 collisions 放在 link 下面
        collision_path = f"{root_prim_path}/{link_name}/collisions"
        col_prim = stage.GetPrimAtPath(collision_path)
        
        if not col_prim.IsValid():
            print(f"Warning: Collision prim not found at {collision_path}!")
            continue
            
        print(f"Applying PhysicsMaterialAPI to {collision_path}...")
        
        # 3. 核心步骤：不创建外部材质节点，直接把“摩擦力属性 (MaterialAPI)”挂载到现在的几何体上！
        # 这就等价于图形界面里 Add -> Physics -> Physics Material 的操作
        physics_material_api = UsdPhysics.MaterialAPI.Apply(col_prim)
        
        # 4. 设置摩擦系数和恢复系数
        physics_material_api.CreateStaticFrictionAttr(1.0)
        physics_material_api.CreateDynamicFrictionAttr(1.0)
        physics_material_api.CreateRestitutionAttr(0.0)
        
    # 5. 保存！
    stage.Save()
    print("Successfully saved USD! The replication issue should be gone, and wheels are high-friction now.")

if __name__ == "__main__":
    import os
    # 定位你的 USD
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_file = os.path.join(current_dir, "source/pipe_robot_lab/pipe_robot_lab/assets/pipe_robot/usd/pipe_robot_rename.usd")
    fix_robot_usd(usd_file)
