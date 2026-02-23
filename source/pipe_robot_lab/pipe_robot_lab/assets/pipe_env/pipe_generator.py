import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import yaml
# from isaaclab.assets import AssetBaseCfg
from dataclasses import dataclass
from typing import List, Tuple
import pickle # 用于保存生成的管道参数以及其次变换矩阵

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config" , "pipe_env.yaml")
STRAIGHT_PIPE_PATH = os.path.join(CURRENT_DIR, "meshes", "stand_pipe.STL")

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
        
        # ? 1. 实例化 GeneratorCfg
        # 使用 .get() 提供默认值或 None，增强健壮性
        gen_data = self._cfg_dict.get("generation", {})
        self.generation = GeneratorCfg(
            mode=gen_data.get("mode", "random"),
            second_mode=gen_data.get("second_mode", "safe"),
            fixed_info=gen_data.get("fixed_info", [])
        )

        # ? 2. 实例化 GeometryCfg
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
        # 采样生成目标管道数目
        num_pipes = np.random.randint(self.pipe.num_pipe_prims[0], self.pipe.num_pipe_prims[1])
        
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
        return combined_mesh
        
        
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
    
    def generate_straight_pipe(self, dia, length , trans = None):
        """
        从原型样板文件(1m直径1m长)生成直管的STL文件, 原始文件中y轴为管长方向
        """
        pipe_mesh = mesh.Mesh.from_file(STRAIGHT_PIPE_PATH)
        # 缩放管道以匹配指定直径和长度
        pipe_mesh.x *= dia
        pipe_mesh.y *= length
        pipe_mesh.z *= dia
        # 可选：应用变换（旋转/平移）
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
    print(f"Loaded Mode: {cfg.generation.mode}")
    print(f"Loaded Dia Range: {cfg.pipe.D_range}")
    
    # 生成管道网格
    combined_mesh = cfg.generate_pipe()
    
    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #1. 绘制世界坐标系原点和轴
    origin = [0, 0, 0]
    length = 1.0 # 坐标轴长度，根据现在的单位是米，设为1米比较合适，或者根据bbox调整
    # X轴 - 红色
    ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='World X')
    # Y轴 - 绿色 
    ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='World Y')
    # Z轴 - 蓝色
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
    plt.show()
