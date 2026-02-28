import os
import glob
import yaml
import random
from dataclasses import dataclass
from typing import List

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from omni.kit.asset_converter import get_instance as get_converter_instance

# =========================
# 路径与配置
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config", "pipe_env.yaml")
MESHES_DIR = os.path.join(CURRENT_DIR, "meshes")
USD_DIR = os.path.join(CURRENT_DIR, "usd")

# -------------------------
# 网格朝向/双面碰撞调试开关
# -------------------------
# True: 强制反转所有面的顶点顺序（用于整体法线方向疑似反了的情况）
FORCE_FLIP_WINDING = False
# True: 为每个面复制一份反向面，得到双面碰撞效果（薄壁单层网格建议开启）
MAKE_COLLISION_TWO_SIDED = True
# True: 若存在 Looks/DefaultMaterial，则直接把物理参数写到该材质上
USE_DEFAULT_MATERIAL_FOR_PHYSICS = True

@dataclass
class PhysicsCfg:
    static_friction_range: List[float] = None
    dynamic_friction_range: List[float] = None
    restitution_range: List[float] = None

def load_physics_cfg(path: str) -> PhysicsCfg:
    """从 YAML 读取摩擦/恢复系数范围。"""
    if not os.path.exists(path):
        print(f"[Warning] Config file not found at {path}, utilizing defaults.")
        return PhysicsCfg([0.8, 1.0], [0.5, 0.5], [0.0, 0.0])
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            phys_data = data.get("physics", {})
            return PhysicsCfg(
                static_friction_range=phys_data.get("static_friction_range", [0.5, 1.0]),
                dynamic_friction_range=phys_data.get("dynamic_friction_range", [0.5, 0.8]),
                restitution_range=phys_data.get("restitution_range", [0.0, 0.1])
            )
    except Exception as e:
        print(f"[Error] Failed to load YAML: {e}")
        return PhysicsCfg()


def _flip_mesh_winding(mesh_prim):
    """反转 Mesh 每个面的顶点顺序（只改变朝向，不改变几何形状）。"""
    mesh_api = UsdGeom.Mesh(mesh_prim)
    counts = mesh_api.GetFaceVertexCountsAttr().Get()
    indices = mesh_api.GetFaceVertexIndicesAttr().Get()
    if not counts or not indices:
        return

    flipped_indices = []
    cursor = 0
    for c in counts:
        face = indices[cursor: cursor + c]
        flipped_indices.extend(list(reversed(face)))
        cursor += c

    mesh_api.GetFaceVertexIndicesAttr().Set(flipped_indices)


def _make_mesh_two_sided(mesh_prim):
    """将每个面追加一个反向面，构造双面碰撞网格。"""
    mesh_api = UsdGeom.Mesh(mesh_prim)
    counts = mesh_api.GetFaceVertexCountsAttr().Get()
    indices = mesh_api.GetFaceVertexIndicesAttr().Get()
    if not counts or not indices:
        return

    reversed_indices = []
    cursor = 0
    for c in counts:
        face = indices[cursor: cursor + c]
        reversed_indices.extend(list(reversed(face)))
        cursor += c

    new_counts = list(counts) + list(counts)
    new_indices = list(indices) + reversed_indices
    mesh_api.GetFaceVertexCountsAttr().Set(new_counts)
    mesh_api.GetFaceVertexIndicesAttr().Set(new_indices)


def _bind_physics_material(mesh_prim, material_path: Sdf.Path):
    """将物理材质绑定到 Mesh（physics:material:binding）。"""
    rel = mesh_prim.CreateRelationship("physics:material:binding", False)
    rel.SetTargets([material_path])


def _get_or_create_physics_material(stage: Usd.Stage, root_prim: Usd.Prim) -> UsdShade.Material:
    """优先复用 DefaultMaterial 作为物理材质，否则创建 PhysicsMaterial。"""
    looks_path = root_prim.GetPath().AppendPath("Looks")
    if not stage.GetPrimAtPath(looks_path):
        UsdGeom.Scope.Define(stage, looks_path)

    if USE_DEFAULT_MATERIAL_FOR_PHYSICS:
        default_mat_path = looks_path.AppendPath("DefaultMaterial")
        default_mat_prim = stage.GetPrimAtPath(default_mat_path)
        if default_mat_prim and default_mat_prim.IsValid():
            return UsdShade.Material(default_mat_prim)

    physics_mat_path = looks_path.AppendPath("PhysicsMaterial")
    return UsdShade.Material.Define(stage, physics_mat_path)

# =========================
# 直接转换为USD文件
# =========================

async def convert_stl_to_usd(input_stl: str, output_usd: str):
    """调用 Isaac 资产转换器：STL -> USD。"""
    converter = get_converter_instance()
    context = omni.kit.asset_converter.AssetConverterContext()

    # 只保留几何体，忽略动画/相机/灯光
    context.ignore_material = False
    context.ignore_animation = True
    context.ignore_camera = True
    context.ignore_light = True
    context.use_meter_as_world_unit = True
    
    print(f"[Converter] Converting {os.path.basename(input_stl)} -> {os.path.basename(output_usd)}")
    task = converter.create_converter_task(input_stl, output_usd, None, context)
    success = await task.wait_until_finished()
    if not success:
        print(f"[Error] Failed to convert {input_stl}")

# =========================
# USD 后处理：设置物理属性以及环境尺度等
# =========================

def modify_usd_physics(usd_path: str, phys_cfg: PhysicsCfg):
    """
    打开转换后的 USD，执行物理后处理：
    1) 设置单位/坐标轴
    2) 生成随机物理材质
    3) 绑定材质并启用碰撞
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[Error] Failed to open stage {usd_path}")
        return
    
    # 统一 USD 场景单位：1.0 代表 1 米，Z 轴朝上
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 按配置随机采样物理参数
    static_friction = random.uniform(phys_cfg.static_friction_range[0], phys_cfg.static_friction_range[1])
    dynamic_friction = random.uniform(phys_cfg.dynamic_friction_range[0], phys_cfg.dynamic_friction_range[1])
    restitution = random.uniform(phys_cfg.restitution_range[0], phys_cfg.restitution_range[1])
    
    # 查找根 Prim。若无默认 Prim，则选择第一个 Mesh/Xform 并设为默认。
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                root_prim = prim
                stage.SetDefaultPrim(root_prim)
                break
    
    if not root_prim:
        print(f"[Warning] Could not find root prim for {usd_path}")
        return

    # 选择一个材质承载物理参数：优先复用 DefaultMaterial
    material = _get_or_create_physics_material(stage, root_prim)
    material_prim = material.GetPrim()
    phys_mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    phys_mat_api.CreateStaticFrictionAttr().Set(static_friction)
    phys_mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat_api.CreateRestitutionAttr().Set(restitution)
    
    # 遍历所有网格：绑定材质 + 启用碰撞
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            # 可选：修正法线方向（通过反转 winding）
            if FORCE_FLIP_WINDING:
                _flip_mesh_winding(prim)

            # 可选：复制反向面以实现双面碰撞
            if MAKE_COLLISION_TWO_SIDED:
                _make_mesh_two_sided(prim)

            # 物理材质绑定：使用 physics:material:binding（而非渲染材质绑定）
            _bind_physics_material(prim, material.GetPath())
            
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            # 使用三角网格碰撞，避免默认凸包把管道“封死”
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr().Set("sdf")  # 禁止自动生成凸包

            # 可视网格设为双面显示
            UsdGeom.Gprim(prim).CreateDoubleSidedAttr(True)
            
    print(f"   -> Physics material path: {material.GetPath()}")
    print(f"   -> Applied Physics: Static={static_friction:.2f}, Dyn={dynamic_friction:.2f}, Rest={restitution:.2f}")
    stage.GetRootLayer().Save()

def run_conversion_pipeline():
    """主流程：扫描增量文件并逐个转换+后处理。"""

    # 1) 检查目录
    if not os.path.exists(MESHES_DIR):
        print(f"[Error] Meshes directory not found: {MESHES_DIR}")
        return
    if not os.path.exists(USD_DIR):
        os.makedirs(USD_DIR)

    # 2) 读取物理配置
    phys_cfg = load_physics_cfg(YAML_PATH)

    # 3) 增量扫描：仅处理 meshes 中有而 usd 中没有的同名文件
    stl_files = sorted(glob.glob(os.path.join(MESHES_DIR, "*.STL")))
    usd_files = glob.glob(os.path.join(USD_DIR, "*.usd"))
    existing_usd_names = {os.path.splitext(os.path.basename(f))[0] for f in usd_files}
    files_to_process = []
    for stl_path in stl_files:
        basename = os.path.splitext(os.path.basename(stl_path))[0]
        if basename not in existing_usd_names:
            files_to_process.append(stl_path)
    if not files_to_process:
        print("[Info] No new STL files to convert. All assets are synchronized.")
        return
    print(f"[Info] Found {len(files_to_process)} new files to convert.")

    # 4) 逐个转换并追加物理属性
    import asyncio
    async def process_queue():
        for stl_path in files_to_process:
            basename = os.path.splitext(os.path.basename(stl_path))[0]
            usd_path = os.path.join(USD_DIR, f"{basename}.usd")
            await convert_stl_to_usd(stl_path, usd_path)
            modify_usd_physics(usd_path, phys_cfg)
    asyncio.run(process_queue())
    print("[Success] All conversions completed.")

if __name__ == "__main__":
    run_conversion_pipeline()
    simulation_app.close()
