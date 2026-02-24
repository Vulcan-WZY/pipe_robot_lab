import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import yaml
# from isaaclab.assets import AssetBaseCfg
from dataclasses import dataclass
from typing import List, Tuple
import json # 用于保存生成的管道参数以及其次变换矩阵
from pathlib import Path

# ==========================================
# 全局配置开关
# ==========================================
ENABLE_VISUALIZATION = True  # 是否在生成时实时显示 3D 网格
VISUALIZATION_PAUSE_TIME = 1.0  # 每个网格显示的停留时间（秒）
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config" , "pipe_env.yaml")
STRAIGHT_PIPE_PATH = os.path.join(CURRENT_DIR, "stand_pipe.STL")

@dataclass
class GeneratorCfg:
    mode: str = "random"  # "random" or "fixed"
    fixed_info: List[List[float]] = None 
    second_mode: str = "safe"  # "random" or "safe"
    total_nums: int = 20
    meshes_path: str = "./meshes"
    usd_path: str = "./usd"
    json_path: str = "./json"

@dataclass
class GeometryCfg:
    """管道几何参数配置"""
    num_pipe_prims: List[int] = None
    D_range: List[float] = None
    L_range_straight: List[float] = None
    D_change_prob: float = 0.1
        
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
        
        # ? 1. 实例化 GeneratorCfg
        # 使用 .get() 提供默认值或 None，增强健壮性
        gen_data = self._cfg_dict.get("generation", {})
        self.generation = GeneratorCfg(
            mode=gen_data.get("mode", "random"),
            second_mode=gen_data.get("second_mode", "safe"),
            fixed_info=gen_data.get("fixed_info", []),
            total_nums=gen_data.get("total_nums", 20),
            meshes_path=gen_data.get("meshes_path", "./meshes"),
            usd_path=gen_data.get("usd_path", "./usd"),
            json_path=gen_data.get("json_path", "./json")
        )

        # ? 2. 实例化 GeometryCfg
        pipe_data = self._cfg_dict.get("pipe", {})
        self.pipe = GeometryCfg(
            num_pipe_prims=pipe_data.get("num_pipe_prims", [3, 6]),
            D_range=pipe_data.get("D_range", [0.2, 0.4]),
            D_change_prob=pipe_data.get("D_change_prob", 0.1),
            L_range_straight=pipe_data.get("L_range_straight", [0.5, 3.0]),
            L_range_bend=pipe_data.get("L_range_bend", [0.3, 0.5]),
            angle_range_bend=pipe_data.get("angle_range_bend", [45, 90]),
            D_step_bend=pipe_data.get("D_step_bend", 0.05),
            L_step_bend=pipe_data.get("L_step_bend", 0.1),
            angle_step_bend=pipe_data.get("angle_step_bend", 10.0),
            probability_bend=pipe_data.get("probability_bend", 0.5)
        )

        # ? 3. 实例化 PhysicsCfg
        phys_data = self._cfg_dict.get("physics", {})
        self.physics = PhysicsCfg(
            static_friction_range=phys_data.get("static_friction_range", [0.8, 1.0]),
            dynamic_friction_range=phys_data.get("dynamic_friction_range", [0.7, 0.8]),
            restitution_range=phys_data.get("restitution_range", [0.0, 0.1])
        )

        # ? 4. 初始化存储生成管道参数的列表
        self.all_generated_pipes = []  # 每个元素是一个 dict，包含管道参数和变换矩阵等信息
        # 从读取到的L_range_bend中按照L_step_bend为步长生成一个L_bend的候选列表(步长不足时保留最后一个值)
        self.bend_L = np.arange(self.pipe.L_range_bend[0], self.pipe.L_range_bend[1] , self.pipe.L_step_bend)
        if self.bend_L[-1] < self.pipe.L_range_bend[1]:
            self.bend_L = np.append(self.bend_L, self.pipe.L_range_bend[1])
        self.bend_D = np.arange(self.pipe.D_range[0], self.pipe.D_range[1] , self.pipe.D_step_bend)
        if self.bend_D[-1] < self.pipe.D_range[1]:
            self.bend_D = np.append(self.bend_D, self.pipe.D_range[1])
        self.bend_angle = np.arange(self.pipe.angle_range_bend[0], self.pipe.angle_range_bend[1] , self.pipe.angle_step_bend)
        if self.bend_angle[-1] < self.pipe.angle_range_bend[1]:
            self.bend_angle = np.append(self.bend_angle, self.pipe.angle_range_bend[1])
    def generate_pipe(self):
        # ! 核心生成管道队列的函数
        # 清空之前生成的管道信息
        self.all_generated_pipes.clear()
        # 采样生成目标管道数目
        num_pipes = np.random.randint(self.pipe.num_pipe_prims[0], self.pipe.num_pipe_prims[1]+1)
        
        # 初始化原始齐次变换矩阵， 初始直管道沿世界坐标系y轴放置，原点为(0,0,0)
        T = np.eye(4)
        pipe_mesh = [];
        
        for i in range(num_pipes): # 管道编号从0 到 num_pipes-1
            # 首先判断需要生成的管道类型
            type = 0
            property_temp = np.random.rand()
            if i == 0 or i == num_pipes-1: # 第一段和最后一段必须是直管
                pass            
            elif self.generation.second_mode == "safe":
                # safe模式， 弯管前后必须为直管
                if self.all_generated_pipes[-1]['info'][1] == 0 and  property_temp < self.pipe.probability_bend: # 前一段是直管， 当前段有概率是弯管
                    type = 1
                else:
                    pass
            elif self.generation.second_mode == "random" and property_temp < self.pipe.probability_bend:
                type = 1
            else:
                pass
                
            # * 进入这里， 说明当前管段是弯管
            if type == 1: 
                # 进入这里， 说明当前管段是弯管
                phi = np.random.uniform(0, 2*np.pi)
                theta = np.random.choice(self.bend_angle) * np.pi / 180.0 
                L = np.random.choice(self.bend_L)
                D = np.random.choice(self.bend_D)
                if self.generation.second_mode == "safe":
                    D = self.all_generated_pipes[-1]['info'][2] # safe模式下弯管的直径必须等于前一段管道的直径
                # 生成前置变换矩阵(绕当前维护的变换矩阵y轴旋转phi角度)
                T_pre = np.array([
                    [np.cos(phi), 0.0, np.sin(phi), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-np.sin(phi), 0.0, np.cos(phi), 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                T = T @ T_pre
                # 生成弯管道网格
                bent_pipe_mesh = self.generate_bent_pipe(diameter=D, radius=L, angle_rad=theta, trans=T)
                pipe_mesh.append(bent_pipe_mesh)
                # 生成管道信息并保存
                info = {
                    'transform': T,
                    'info': [i, type, D, L, phi, theta]
                }
                self.all_generated_pipes.append(info)   
                # 生成后置变换矩阵， 并更新T
                T_post = np.array([
                    [1.0 , 0.0 , 0.0 , 0.0],
                    [0.0 , np.cos(theta) , -np.sin(theta) , L * np.sin(theta)],
                    [0.0 , np.sin(theta) , np.cos(theta) , L * (1 - np.cos(theta))],
                    [0.0 , 0.0 , 0.0 , 1.0]
                ])
                T = T @ T_post
            # * 进入这里， 说明当前管段是直管
            elif type == 0: 
                # 进入这里， 说明当前管段是直管
                d = np.random.uniform(self.pipe.D_range[0], self.pipe.D_range[1])
                if np.random.rand() > self.pipe.D_change_prob and i > 0: # 有一定概率保持直径不变， 增加管道的连续性
                    d = self.all_generated_pipes[-1]['info'][2]
                # 安全模式限制管径
                if self.generation.second_mode == "safe" and i > 0:
                    if self.all_generated_pipes[-1]['info'][1] == 1: 
                        d = self.all_generated_pipes[-1]['info'][2]
                l = np.random.uniform(self.pipe.L_range_straight[0], self.pipe.L_range_straight[1])
                
                # 生成直管道网格
                pipe_mesh.append(
                    self.generate_straight_pipe(dia= d, length=l, trans=T)
                )
                # 生成管道信息并保存
                info = {
                    'transform': T,
                    # 管道编号， 管道类型， 管道直径， 管道长度， 前置偏转， 后置偏转                    
                    'info': [i, type , d , l , 0.0 , 0.0]
                }
                self.all_generated_pipes.append(info)
                # 生成后置变换矩阵， 并更新T
                T_post = np.array([
                    [1.0 , 0.0 , 0.0 , 0.0],
                    [0.0 , 1.0 , 0.0 , l  ],
                    [0.0 , 0.0 , 1.0 , 0.0],
                    [0.0 , 0.0 , 0.0 , 1.0]
                ])
                T = T @ T_post
        # * 将所有的管道网格进行合并
        combined_mesh = mesh.Mesh(np.concatenate([m.data for m in pipe_mesh]))
        return combined_mesh , self.all_generated_pipes
        
        
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
    
    def generate_straight_pipe(self, dia, length , trans = None, num_points=35):
        """
        Procedurally generate a straight pipe mesh (cylinder) along the Y-axis.
        """
        # Create vertices
        vertices = []
        # Bottom circle (y=0) and Top circle (y=length)
        phi = np.linspace(0, 2 * np.pi, num_points)
        r = dia / 2.0
        
        # Vertices for the side surface
        for j in range(num_points):
            ph = phi[j]
            px = r * np.sin(ph)
            pz = r * np.cos(ph) # align to match bent pipe's start orientation
            # Bottom vertex
            vertices.append([px, 0.0, pz])
            # Top vertex
            vertices.append([px, length, pz])
            
        # Add center points for caps
        vertices.append([0.0, 0.0, 0.0])      # Bottom center
        idx_bottom_center = len(vertices) - 1
        vertices.append([0.0, length, 0.0])   # Top center
        idx_top_center = len(vertices) - 1
        
        # Create faces
        faces = []
        for j in range(num_points):
            # Side faces (2 triangles per segment)
            # Indices: 2*j (bottom), 2*j+1 (top)
            # Next indices: 2*((j+1)%num_points), 2*((j+1)%num_points)+1
            
            p1 = 2 * j
            p2 = 2 * j + 1
            p3 = 2 * ((j + 1) % num_points) + 1
            p4 = 2 * ((j + 1) % num_points)
            
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
            
            # Bottom cap
            faces.append([idx_bottom_center, 2 * ((j + 1) % num_points), 2 * j])
            
            # Top cap
            faces.append([idx_top_center, 2 * j + 1, 2 * ((j + 1) % num_points) + 1])
            
        # Create mesh
        data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
        pipe_mesh = mesh.Mesh(data)
        for i, f in enumerate(faces):
            for j in range(3):
                pipe_mesh.vectors[i][j] = vertices[f[j]]
                
        if trans is not None:
            pipe_mesh.transform(trans)
        return pipe_mesh
    
    def generate_bent_pipe(self, diameter, radius, angle_rad, trans = None, num_step=5, num_points=35):
        """
        生成弯管的STL文件
        :param diameter: 弯管直径
        :param radius: 弯曲半径（中心线半径）
        :param angle_rad: 弯曲角度（圆心角，弧度）
        :param num_step: 中心线的角度步长数（沿管长方向的分段数）
        :param num_points: 横截面的点数
        """
        # 转换角度为弧度
        angle = angle_rad
        num_segments = int(angle * 180.0 / np.pi) // ( num_step) + 1  # 根据角度和步长计算分段数
        
        # 角度采样
        theta = np.linspace(0, angle, num_segments)  # 沿管长方向
        phi = np.linspace(0, 2 * np.pi, num_points)   # 沿截面圆周方向
        
        # 创建顶点
        vertices = []
        r_inner = diameter / 2
        for i in range(num_segments):
            th = theta[i]  # 圆心角
            for j in range(num_points):
                ph = phi[j]
                # 目标：起点在 (0,0,0)，中心线半径 R，终点中心在 (0, R, R)
                # 截面中心点轨迹：(0, radius * sin(th), radius * (1 - cos(th)))
                px = r_inner * np.sin(ph)
                py = (radius + r_inner * np.cos(ph)) * np.sin(th)
                pz = radius - (radius + r_inner * np.cos(ph)) * np.cos(th)
                vertices.append([px, py, pz])
                
        # 添加两个中心点用于封口
        vertices.append([0.0, 0.0, 0.0])  # 起点中心
        idx_start_center = len(vertices) - 1
        vertices.append([0.0, radius * np.sin(angle), radius * (1 - np.cos(angle))])  # 终点中心
        idx_end_center = len(vertices) - 1
        
        # 创建面（三角形）
        faces = []
        for i in range(num_segments - 1):
            for j in range(num_points):
                # 四个顶点的索引
                p1 = i * num_points + j
                p2 = (i + 1) * num_points + j
                p3 = (i + 1) * num_points + (j + 1) % num_points
                p4 = i * num_points + (j + 1) % num_points
                
                # 使用两个三角形组成一个矩形面
                faces.append([p1, p2, p3])
                faces.append([p1, p3, p4])
        
        # 端面封口:
        for j in range(num_points):
            # 起点端面 (索引 0 到 num_points-1)
            faces.append([idx_start_center, (j + 1) % num_points, j])
            # 终点端面 (索引最后一次循环生成的点)
            p_last_base = (num_segments - 1) * num_points
            faces.append([idx_end_center, p_last_base + j, p_last_base + (j + 1) % num_points])
        
        # 创建STL网格
        # 初始化数据结构
        data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
        pipe_mesh = mesh.Mesh(data)
        for i, f in enumerate(faces):
            for j in range(3):
                pipe_mesh.vectors[i][j] = vertices[f[j]]
        
        if trans is not None:
            pipe_mesh.transform(trans)
        
        return pipe_mesh

# 单元测试 (仅当直接运行此文件时执行)
if __name__ == "__main__":
    cfg = PipeEnvGenerator()
    MESHES_PATH = os.path.join(CURRENT_DIR, cfg.generation.meshes_path)
    JSON_PATH = os.path.join(CURRENT_DIR, cfg.generation.json_path)
    total_nums = cfg.generation.total_nums
    
    id_length = 6 # 生成文件的保存ID号位数，如1号文件为000001
    # 获取当前已经存在的文件数量，以免编号重复
    mesh_dir = Path(MESHES_PATH)
    json_dir = Path(JSON_PATH)
    mesh_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在，避免首次运行报错
    json_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在，避免首次运行报错
    existing_ids = (int(p.stem) for p in mesh_dir.glob("*.STL") if p.stem.isdigit())
    start_id = max(existing_ids, default=0) + 1 # 这里不用+1 因为对应路径下放的有一个stand_pipe.STL
    
    
    # 打印一些配置的信息
    print("Generation Mode:", cfg.generation.mode)
    print("Total Pipes to Generate:", total_nums)
        
    print("Meshes Path:", MESHES_PATH)
    print("JSON Path:", JSON_PATH)
    print("Start ID for Saving Files:", start_id)
    
    # 如果开启了可视化，开启交互模式
    if ENABLE_VISUALIZATION:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # 生成管道网格
    for i in range(total_nums):
        [combined_mesh , all_pipes_info] = cfg.generate_pipe()
        # 保存网格和信息
        filename = f"{i + start_id:0{id_length}d}"
        mesh_path = os.path.join(MESHES_PATH, f"{filename}.STL")
        json_path = os.path.join(JSON_PATH, f"{filename}.json")
        combined_mesh.save(mesh_path)
        
        # 自定义一个编码器，用于处理 NumPy 数组
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist() # 将 numpy 数组转为普通列表
                return super().default(obj)
                
        with open(json_path, 'w') as f:
            json.dump(all_pipes_info, f, cls=NumpyEncoder, indent=2)
            
        # ---------------------------------------------------------
        # 实时可视化与信息打印
        # ---------------------------------------------------------
        if ENABLE_VISUALIZATION:
            print(f"\n[{i+1}/{total_nums}] Generated Pipe: {filename}")
            for idx, pipe_info in enumerate(all_pipes_info):
                print(f"  Segment {idx}: Type={'Bend' if pipe_info['info'][1]==1 else 'Straight'}, "
                    f"D={pipe_info['info'][2]:.3f}, L={pipe_info['info'][3]:.3f}")
                print(f"  Transform:\n{pipe_info['transform']}")
            
            ax.clear() # 清除上一帧
            
            # 1. 绘制世界坐标系原点和轴
            length = 1.0 
            ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='World X')
            ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='World Y')
            ax.quiver(0, 0, 0, 0, 0, length, color='b', arrow_length_ratio=0.1, linewidth=2, label='World Z')

            poly_collection = art3d.Poly3DCollection(combined_mesh.vectors)
            poly_collection.set_edgecolor('k')
            poly_collection.set_alpha(0.6)
            ax.add_collection3d(poly_collection)
            
            # 设置坐标轴显示范围，防止图形被压缩
            max_range = np.array([combined_mesh.x.max()-combined_mesh.x.min(), 
                                combined_mesh.y.max()-combined_mesh.y.min(), 
                                combined_mesh.z.max()-combined_mesh.z.min()]).max() / 2.0
            
            mid_x = (combined_mesh.x.max()+combined_mesh.x.min()) * 0.5
            mid_y = (combined_mesh.y.max()+combined_mesh.y.min()) * 0.5
            mid_z = (combined_mesh.z.max()+combined_mesh.z.min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Pipe ID: {filename}")
            
            plt.draw()
            plt.pause(VISUALIZATION_PAUSE_TIME) # 暂停指定时间
            
    if ENABLE_VISUALIZATION:
        plt.ioff() # 关闭交互模式
        plt.show() # 保持最后一个窗口打开
    else:
        print(f"\nSuccessfully generated {total_nums} pipes.")
