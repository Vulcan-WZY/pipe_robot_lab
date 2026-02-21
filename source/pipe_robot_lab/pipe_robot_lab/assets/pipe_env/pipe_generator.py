import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import yaml
# from isaaclab.assets import AssetBaseCfg
from dataclasses import dataclass
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config" , "pipe_env.yaml")

@dataclass
class GeneratorCfg:
    mode: str = "random"  # "random" or "fixed"
    fixed_info: List[List[float]] = None 
    second_mode: str = "safe"  # "random" or "safe"

@dataclass
class GeometryCfg:
    """管道几何参数配置"""
    num_pipe_prims: List[int] = None
    D_range: List[float] = None
    L_range_straight: List[float] = None
    L_range_bend: List[float] = None
    angle_range_bend: List[float] = None
    D_step_bend: float = 0.05
    L_step_bend: float = 0.1
    angle_step_bend: float = 10.0
    probability_bend: float = 0.5

@dataclass
class PhysicsCfg:
    static_friction_range: List[float] = None # 静摩擦系数范围
    dynamic_friction_range: List[float] = None # 动摩擦系数范围
    restitution_range: List[float] = None # 恢复系数范围

class PipeEnvGenerator:
    def __init__(self, yaml_path: str = YAML_PATH):
        self._cfg_dict = self._load_yaml(yaml_path)
        
        # 1. 实例化 GeneratorCfg
        # 使用 .get() 提供默认值或 None，增强健壮性
        gen_data = self._cfg_dict.get("generation", {})
        self.generation = GeneratorCfg(
            mode=gen_data.get("mode", "random"),
            second_mode=gen_data.get("second_mode", "safe"),
            fixed_info=gen_data.get("fixed_info", [])
        )

        # 2. 实例化 GeometryCfg
        pipe_data = self._cfg_dict.get("pipe", {})
        self.pipe = GeometryCfg(
            num_pipe_prims=pipe_data.get("num_pipe_prims", [3, 6]),
            D_range=pipe_data.get("D_range", [0.2, 0.4]),
            L_range_straight=pipe_data.get("L_range_straight", [0.5, 3.0]),
            L_range_bend=pipe_data.get("L_range_bend", [0.3, 0.5]),
            angle_range_bend=pipe_data.get("angle_range_bend", [45, 90]),
            D_step_bend=pipe_data.get("D_step_bend", 0.05),
            L_step_bend=pipe_data.get("L_step_bend", 0.1),
            angle_step_bend=pipe_data.get("angle_step_bend", 10.0),
            probability_bend=pipe_data.get("probability_bend", 0.5)
        )

        # 3. 实例化 PhysicsCfg
        phys_data = self._cfg_dict.get("physics", {})
        self.physics = PhysicsCfg(
            static_friction_range=phys_data.get("static_friction_range", [0.8, 1.0]),
            dynamic_friction_range=phys_data.get("dynamic_friction_range", [0.7, 0.8]),
            restitution_range=phys_data.get("restitution_range", [0.0, 0.1])
        )
    
    def _load_yaml(self, path: str) -> dict:
        if not os.path.exists(path):
            print(f"[Warning] Config file not found at {path}, utilizing defaults.")
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Error] Failed to load YAML: {e}")
            return {}

# 单元测试 (仅当直接运行此文件时执行)
if __name__ == "__main__":
    cfg = PipeEnvGenerator()
    print(f"Loaded Mode: {cfg.generation.mode}")
    print(f"Loaded Dia Range: {cfg.pipe.D_range}")
