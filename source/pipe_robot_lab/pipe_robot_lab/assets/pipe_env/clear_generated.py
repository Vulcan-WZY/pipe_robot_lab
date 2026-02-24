import os
import yaml
from dataclasses import dataclass
from typing import List
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config" , "pipe_env.yaml")
STRAIGHT_PIPE_PATH = os.path.join(CURRENT_DIR, "meshes", "stand_pipe.STL")

@dataclass
class GeneratorCfg:
    mode: str = "random"  # "random" or "fixed"
    fixed_info: List[List[float]] = None 
    second_mode: str = "safe"  # "random" or "safe"
    total_nums: int = 20
    meshes_path: str = "./meshes"
    json_path: str = "./json"

class PipeEnvCleaner:
    def __init__(self, yaml_path: str = YAML_PATH):
        self._cfg_dict = self._load_yaml(yaml_path)
        
        # ? 1. 实例化 GeneratorCfg
        gen_data = self._cfg_dict.get("generation", {})
        self.generation = GeneratorCfg(
            mode=gen_data.get("mode", "random"),
            second_mode=gen_data.get("second_mode", "safe"),
            fixed_info=gen_data.get("fixed_info", []),
            total_nums=gen_data.get("total_nums", 20),
            meshes_path=gen_data.get("meshes_path", "./meshes"),
            json_path=gen_data.get("json_path", "./json")
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
    cfg = PipeEnvCleaner()
    MESHES_PATH = os.path.join(CURRENT_DIR, cfg.generation.meshes_path)
    JSON_PATH = os.path.join(CURRENT_DIR, cfg.generation.json_path)
    
    mesh_dir = Path(MESHES_PATH)
    json_dir = Path(JSON_PATH)
    
    print(f"Target Meshes Path: {MESHES_PATH}")
    print(f"Target JSON Path: {JSON_PATH}")
    
    # 1. 清理 JSON 文件
    json_count = 0
    if json_dir.exists():
        for json_file in json_dir.glob("*.json"):
            try:
                json_file.unlink()
                json_count += 1
            except Exception as e:
                print(f"Failed to delete {json_file}: {e}")
    
    # 2. 清理 STL 文件 (保留 stand_pipe.STL)
    stl_count = 0
    if mesh_dir.exists():
        for stl_file in mesh_dir.glob("*.STL"):
            # 检查是否是保留文件
            if stl_file.name == "stand_pipe.STL":
                print(f"Skipping template file: {stl_file.name}")
                continue
            
            try:
                stl_file.unlink()
                stl_count += 1
            except Exception as e:
                print(f"Failed to delete {stl_file}: {e}")
                
    print(f"\nCleanup Complete!")
    print(f"Deleted {json_count} JSON files.")
    print(f"Deleted {stl_count} STL files.")
