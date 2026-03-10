# ===========
# Date: 2026-03-02 11:41
# Author: Vulcan
# LastEditTime: 2026-03-10 23:56
# Description: 用于存放一些计算与给定管道进行交互的函数，处理点或机器人与完整管路的关系
# ==========

import torch

# 判断一个世界坐标系下的三维坐标点在整个管道系统中的哪个管道段上，返回管元编号，该点在管元下的相对坐标，以及对应的轴线点长度占比
def is_point_on_segment(point: torch.Tensor, trans_raw: torch.Tensor, info_raw: torch.Tensor) -> torch.Tensor:
    """
    point: [3] 世界坐标系下的点坐标
    trans_raw: [N,4,4] 每个管道段的齐次变换矩阵，描述管道段在世界坐标系下的位置和姿态
    info_raw: [N,6] 每个管道段的描述参数
    """
    
def is_point_on_pipe(point: torch.Tensor, trans: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
    """
    point: [3] 世界坐标系下的点坐标
    trans: [4,4] 某一个具体的管道元件的齐次变换矩阵，描述管道段在世界坐标系下的位置和姿态
    info: [6] 某一个具体的管道元件的描述参数
    """
    
    # p_i = (T^W_i)^-1 * p_world
    point_h = torch.cat([point, torch.ones(1,dtype=point.dtype, device=point.device)], dim=0) # [4]
    point_local_h = torch.linalg.inv(trans) @ point_h # [4]
    point_local = point_local_h[:3] # [3] 取前三维为局部坐标
    pipe_aix_dis = 0.0
    # info: 管道编号， 管道类型， 管道直径， 管道长度， 前置偏转， 后置偏转    
    type = info[1]
    if type == 0: # 直管段
        pipe_aix_dis = point_local[1] / info[3]
        pass
    elif type == 1: # 弯管
        pass
    return torch.cat([point_local,torch.tensor([pipe_aix_dis], dtype=point.dtype, device=point.device)], dim=0) # [4]

# 定位到生成器产生的管道路径
# from pipe_robot_lab.assets.pipe_env.pipe_generator import CURRENT_DIR
import os
import glob
import random , json
CURRENT_DIR = "/home/vulcan/Academic/pipe_robot_lab/source/pipe_robot_lab/pipe_robot_lab/assets/pipe_env"
MESHES_DIR = os.path.join(CURRENT_DIR, "usd")
JSON_DIR = os.path.join(CURRENT_DIR, "json")
# 获取所有生成的 STL 文件 (排除模板文件 stand_pipe.STL)
all_pipe_stls = [p for p in glob.glob(os.path.join(MESHES_DIR, "*.usd"))]
# 随机选择一个管道 (如果文件夹为空则给个默认空字符串防报错)
SELECTED_PIPE_STL = random.choice(all_pipe_stls) if all_pipe_stls else ""
# 推导出对应的 JSON 文件路径
SELECTED_PIPE_JSON = SELECTED_PIPE_STL.replace("usd", "json").replace(".usd", ".json") if SELECTED_PIPE_STL else ""

# 测试用主程序
if __name__ == "__main__":
    # * 读取并挂载本轮随机加载的JSON管道描述文件
    if os.path.exists(SELECTED_PIPE_JSON):
        with open(SELECTED_PIPE_JSON, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        transform_list = [item["transform"] for item in raw_data]
        info_list = [item["info"] for item in raw_data]
        
        # 直接转换为Tensor， PyTorch会自动推断并形成对应形状
        pipe_transform = torch.tensor(transform_list, dtype=torch.float32)  # [N, 4, 4]
        pipe_info = torch.tensor(info_list, dtype=torch.float32)            # [N, 6]
        print(f"[INFO] Pipe Transform Tensor Shape: {pipe_transform.shape}")
        print(f"[INFO] Pipe Info Tensor Shape: {pipe_info.shape}")
    else:
        # Shape: [1, 4, 4]
        pipe_transform = torch.eye(4).unsqueeze(0)  
        # Shape: [1, 6]
        pipe_info = torch.tensor([[0, 0, 0.36, 1.0, 0.0, 0.0]], dtype=torch.float32) 
        print(f"[WARNING] Could not find JSON info at: {SELECTED_PIPE_JSON}")
    
    # 使用第二个管道测试相对坐标转换
    id = 1
    print(is_point_on_pipe(torch.tensor([0.5, 0.0, 0.0]), pipe_transform[id], pipe_info[id]))



