# ===========
# Date: 2026-03-02 11:41
# Author: Vulcan
# LastEditTime: 2026-03-13 10:28
# Description: 用于存放一些计算与给定管道进行交互的函数，处理点或机器人与完整管路的关系
# ==========

import torch
import math
import time

def is_point_on_pipe(points: torch.Tensor, trans_raw: torch.Tensor, info_raw: torch.Tensor) -> torch.Tensor:
    """
    points: [P,3] 世界坐标系下的点坐标
    trans_raw: [N,4,4] 每个管道段的齐次变换矩阵，描述管道段在世界坐标系下的位置和姿态
    info_raw: [N,6] 每个管道段的描述参数
    return: [P,5] 五维向量，包含以下信息：对应点在对应管段的相对坐标xyz, 轴向距离占比，径向距离占比
    """
    if points.dim() == 1:
        points = points.unsqueeze(0)

    device = points.device
    dtype = points.dtype
    num_points = points.size(0)
    best_radial = torch.full((num_points,), float("inf"), dtype=dtype, device=device)
    best_out = torch.zeros((num_points, 5), dtype=dtype, device=device)

    inv_trans = torch.linalg.inv(trans_raw)
    for i in range(info_raw.size(0)):
        seg_out = is_point_on_segment(points, inv_trans[i], info_raw[i, :])
        axial = seg_out[:, 3]
        radial = seg_out[:, 4]
        valid = (axial >= 0.0) & (axial <= 1.0) & (radial >= 1.0)
        better = valid & (radial < best_radial)
        if better.any():
            seg_out = seg_out.clone()
            seg_out[:, 3] = seg_out[:, 3] + i
            best_radial = torch.where(better, radial, best_radial)
            best_out = torch.where(better.unsqueeze(1), seg_out, best_out)

    no_hit = torch.isinf(best_radial)
    if no_hit.any():
        best_out = torch.where(no_hit.unsqueeze(1), torch.zeros_like(best_out), best_out)
    return best_out
    
    
# 判断一个世界坐标系下的三维坐标点在整个管道系统中的哪个管道段上，返回管元编号，该点在管元下的相对坐标，以及对应的轴线点长度占比
def is_point_on_segment(points: torch.Tensor, trans_inv: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
    """
    points: [P,3] 世界坐标系下的点坐标
    trans_inv: [4,4] 某一个具体的管道元件的齐次变换矩阵的逆
    info: [6] 某一个具体的管道元件的描述参数
    return: [P,5] 五维向量
    """
    if points.dim() == 1:
        points = points.unsqueeze(0)

    device = points.device
    dtype = points.dtype

    # p_i = (T^W_i)^-1 * p_world
    ones = torch.ones((points.size(0), 1), dtype=dtype, device=device)
    points_h = torch.cat([points, ones], dim=1)  # [P,4]
    points_local_h = points_h @ trans_inv.T  # [P,4]
    points_local = points_local_h[:, :3]  # [P,3]

    # info: 管道编号， 管道类型， 管道直径， 管道长度， 前置偏转， 后置偏转
    type_id = int(info[1].item())
    D = info[2]
    L = info[3]
    theta = info[5]

    axial = torch.zeros((points.size(0),), dtype=dtype, device=device)
    radial = torch.full((points.size(0),), -1.0, dtype=dtype, device=device)

    if type_id == 0:
        axial = points_local[:, 1] / L
        valid = (axial >= 0.0) & (axial <= 1.0)
        radial_val = torch.sqrt(points_local[:, 0] ** 2 + points_local[:, 2] ** 2) / D * 2.0
        radial = torch.where(valid, radial_val, radial)
    elif type_id == 1:
        theta_p = torch.atan2(points_local[:, 1], L - points_local[:, 2])
        axial = theta_p / theta
        valid = (axial >= 0.0) & (axial <= 1.0)
        target = torch.stack(
            [
                torch.zeros_like(theta_p),
                L * torch.sin(theta_p),
                L * (1.0 - torch.cos(theta_p)),
            ],
            dim=1,
        )
        radial_val = torch.norm(target - points_local, dim=1) / D * 2.0
        radial = torch.where(valid, radial_val, radial)

    return torch.cat([points_local, axial.unsqueeze(1), radial.unsqueeze(1)], dim=1)

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
    
    # 使用单个管道测试相对坐标转换 (批量形式)
    seg_id = 0
    test_points = torch.tensor(
        [
            [0.0, 1.2 + 1.5, 0.5],
            [0.2, 0.4, 0.1],
            [-0.3, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    seg_out = is_point_on_segment(test_points, torch.linalg.inv(pipe_transform[seg_id]), pipe_info[seg_id])
    print("[DEBUG] is_point_on_segment batch output:")
    print(seg_out)

    # 使用整段管道完成主函数测试
    t0 = time.perf_counter()
    result = is_point_on_pipe(test_points, pipe_transform, pipe_info)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"[PERF] is_point_on_pipe elapsed: {elapsed_ms:.3f} ms")
    print("\n================= Query Result =================")
    # result = [x_local, y_local, z_local, (seg_id + axis_ratio), radial_ratio]
    for idx in range(test_points.size(0)):
        point = test_points[idx]
        out = result[idx]
        print(f"world point            : {point.tolist()}")
        if torch.allclose(out, torch.zeros(5, dtype=out.dtype, device=out.device)):
            print("status                 : NOT_ON_PIPE")
            print("segment_id             : -1")
            print("local_xyz              : [0.0, 0.0, 0.0]")
            print("axial_ratio            : 0.0")
            print("radial_ratio           : 0.0")
        else:
            local_xyz = out[:3]
            seg_axis = out[3].item()
            hit_seg_id = int(math.floor(seg_axis))
            axial_ratio = seg_axis - hit_seg_id
            radial_ratio = out[4].item()

            print("status                 : ON_PIPE")
            print(f"segment_id             : {hit_seg_id}")
            print(f"local_xyz              : [{local_xyz[0].item():.6f}, {local_xyz[1].item():.6f}, {local_xyz[2].item():.6f}]")
            print(f"axial_ratio            : {axial_ratio:.6f}")
            print(f"radial_ratio           : {radial_ratio:.6f}")

            if 0 <= hit_seg_id < pipe_info.size(0):
                info = pipe_info[hit_seg_id]
                print(f"segment_info[id,type,D,L,phi,theta]: {info.tolist()}")
        print("---------------------------------------------")
    print("================================================\n")


