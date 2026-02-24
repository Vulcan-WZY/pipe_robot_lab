import os
import glob
import time
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Isaac Sim imports
from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from omni.kit.asset_converter import get_instance as get_converter_instance

# ==========================================
# Configuration Loading
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config", "pipe_env.yaml")
MESHES_DIR = os.path.join(CURRENT_DIR, "meshes")
USD_DIR = os.path.join(CURRENT_DIR, "usd")

@dataclass
class PhysicsCfg:
    static_friction_range: List[float] = None # 静摩擦系数范围
    dynamic_friction_range: List[float] = None # 动摩擦系数范围
    restitution_range: List[float] = None # 恢复系数范围

def load_physics_cfg(path: str) -> PhysicsCfg:
    if not os.path.exists(path):
        print(f"[Warning] Config file not found at {path}, utilizing defaults.")
        return PhysicsCfg([0.5, 0.5], [0.5, 0.5], [0.0, 0.0])
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

# ==========================================
# Conversion Logic
# ==========================================

async def convert_stl_to_usd(input_stl: str, output_usd: str):
    converter = get_converter_instance()
    context = omni.kit.asset_converter.AssetConverterContext()
    # Configure context if needed (e.g. merge materials, etc.)
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

def modify_usd_physics(usd_path: str, phys_cfg: PhysicsCfg):
    """
    Open the converted USD, add a PhysicsMaterial with random properties,
    and bind it to the mesh.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[Error] Failed to open stage {usd_path}")
        return
    
    # Force strict unit and axis settings
    UsdGeom.SetStageMetersPerUnit(stage, 1.0) 
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Randomize values
    static_friction = random.uniform(phys_cfg.static_friction_range[0], phys_cfg.static_friction_range[1])
    dynamic_friction = random.uniform(phys_cfg.dynamic_friction_range[0], phys_cfg.dynamic_friction_range[1])
    restitution = random.uniform(phys_cfg.restitution_range[0], phys_cfg.restitution_range[1])
    
    # 1. Define Physics Material
    # Create a wrapper path for materials
    # We find a valid scope for the material
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        # If no default prim, try to find the first Mesh or Xform
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                root_prim = prim
                stage.SetDefaultPrim(root_prim) # Critical: Set default prim for referencing
                break
    
    if not root_prim:
        print(f"[Warning] Could not find root prim for {usd_path}")
        return

    # Create a unique material path
    # Ensure we use a valid absolute path. If the root prim is directly under absolute root, parent is '/'.
    parent_path = root_prim.GetPath().GetParentPath()
    if parent_path == Sdf.Path.emptyPath:
        parent_path = "/"
        
    material_path = f"{parent_path}/PhysicsMaterial".replace("//", "/")
    # Define a UsdShade Material
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    
    # Apply Physics Material API
    phys_mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    phys_mat_api.CreateStaticFrictionAttr().Set(static_friction)
    phys_mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat_api.CreateRestitutionAttr().Set(restitution)
    
    # 2. Bind Material to all Meshes
    # Iterate over all prims and bind if it is a Mesh
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            # Bind the material use UsdShade
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            
            # Ensure it has Collision API so it acts as a static collider
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            # CRITICAL: Set Collision Approximation to "none" (Triangle Mesh)
            # Default is often "convexHull" which closes the pipe opening.
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr().Set("none")
            
            # --- FIX: Enable Double-Sided Collision ---
            # Since the pipe wall is a single layer of triangles without thickness,
            # we must ensure PhysX treats it as a double-sided surface to prevent 
            # objects inside from tunneling out.
            # Usually handled by the Visual Gprim's doubleSided attribute, 
            # but usually PhysX Mesh Collision reads this.
            
            # 1. Set standard USD doubleSided attribute
            prim.GetAttribute("doubleSided").Set(True)
            
            # We do NOT apply RigidBodyAPI because these are likely static environment obstacles,
            # or if they are dynamic, that is handled by the spawner configuration.
            
    print(f"   -> Applied Physics: Static={static_friction:.2f}, Dyn={dynamic_friction:.2f}, Rest={restitution:.2f}")
    
    stage.GetRootLayer().Save()

def run_conversion_pipeline():
    # 1. Setup Directories
    if not os.path.exists(MESHES_DIR):
        print(f"[Error] Meshes directory not found: {MESHES_DIR}")
        return
    if not os.path.exists(USD_DIR):
        os.makedirs(USD_DIR)
        
    # 2. Load Config
    phys_cfg = load_physics_cfg(YAML_PATH)
    
    # 3. Scan Files
    stl_files = sorted(glob.glob(os.path.join(MESHES_DIR, "*.STL")))
    usd_files = glob.glob(os.path.join(USD_DIR, "*.usd"))
    
    existing_usd_names = {os.path.splitext(os.path.basename(f))[0] for f in usd_files}
    
    files_to_process = []
    
    for stl_path in stl_files:
        basename = os.path.splitext(os.path.basename(stl_path))[0]
        # Check alignment (filename matching)
        if basename not in existing_usd_names:
            files_to_process.append(stl_path)
            
    if not files_to_process:
        print("[Info] No new STL files to convert. All assets are synchronized.")
        return

    print(f"[Info] Found {len(files_to_process)} new files to convert.")
    
    # 4. Process Loop
    import asyncio
    
    async def process_queue():
        for stl_path in files_to_process:
            basename = os.path.splitext(os.path.basename(stl_path))[0]
            usd_path = os.path.join(USD_DIR, f"{basename}.usd")
            
            # Convert
            await convert_stl_to_usd(stl_path, usd_path)
            
            # Modify Physics
            modify_usd_physics(usd_path, phys_cfg)
            
    asyncio.run(process_queue())
    print("[Success] All conversions completed.")

if __name__ == "__main__":
    run_conversion_pipeline()
    simulation_app.close()
