# ===========
# Date: 2026-03-02 11:41
# Author: Vulcan
# LastEditTime: 2026-03-19 12:58
# Description: 用于存放一些计算与给定管道进行交互的函数，处理点或机器人与完整管路的关系
# ==========

import torch
import math
import time

import pipe_robot_lab.funcs.pose_utils as pose_utils

def get_cached_pipe_info(env, asset_cfg) -> torch.Tensor:
    """
    带有单帧缓存机制的管元匹配查询。
    保证在同一个物理仿真步（Step）内，无论 Reward、Termination、Observation 调用多少次，真实计算只发生1次。
    返回: [num_envs, 5] (x_local, y_local, z_local, seg_id + axial_ratio, radial_ratio)
    """
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze(1)
    
    # 1. 创建基于帧/步数的数据缓存防线
    current_step = getattr(env, "common_step_count", 0) 
    
    if not hasattr(env, "_pipe_state_cache"):
        env._pipe_state_cache = {"step": -1, "data": {}}
        
    if env._pipe_state_cache["step"] != current_step:
        env._pipe_state_cache["step"] = current_step
        env._pipe_state_cache["data"].clear() 
        
    cache_key = f"{asset_cfg.name}_{asset_cfg.body_names[0]}"
    
    if cache_key in env._pipe_state_cache["data"]:
        return env._pipe_state_cache["data"][cache_key] 
        
    trans_inv = torch.as_tensor(env.cfg.pipe_transform_inv, dtype=torch.float32, device=env.device)
    info = torch.as_tensor(env.cfg.pipe_info, dtype=torch.float32, device=env.device)
    
    out = is_point_on_pipe(positions, trans_inv, info, already_inverted=True)
    env._pipe_state_cache["data"][cache_key] = out
    
    return out

def get_target_relative_pose(env, asset_cfg) -> torch.Tensor:
    """
    env: ManagerBasedRLEnv
    asset_cfg: SceneEntityCfg 指定包含待评估目标（如机器人后半段或前半段）的资产位姿
    return: angle_diffs [num_envs, 2] 返回机器人在目标位姿下，分别绕目标坐标系 x 轴和 z 轴的角度差值（弧度）。
    """
    device = env.device
    num_envs = env.num_envs
    
    # 获取资产的四元数（用于后续姿态计算）
    asset = env.scene[asset_cfg.name]
    quats = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze(1)
    dtype = quats.dtype
    
    device = quats.device
    
    trans_raw = torch.as_tensor(env.cfg.pipe_transform_inv, dtype=torch.float32, device=device)
    info_raw = torch.as_tensor(env.cfg.pipe_info, dtype=torch.float32, device=device)
    already_inverted = True

    # * 1. 批量获取 (使用缓存的相对坐标进度信息)
    relative_info = get_cached_pipe_info(env, asset_cfg)
    # relative_info:  [num_envs , 5]
    
    # * 2. 从匹配结果中提取局部坐标与进度, 解析所述的管元以及几何参数
    local_xyz = relative_info[:,0:3]
    seg_axis = relative_info[: , 3]
    # 使用掩码处理未命中管网的情况防止越界
    valid_mask = seg_axis >= 0.0
    seg_ids = torch.where(valid_mask, torch.floor(seg_axis), torch.zeros_like(seg_axis)).long()
    axial_ratio = torch.where(valid_mask, seg_axis - seg_ids, torch.zeros_like(seg_axis))
    
    # 批量提取当前环境对应的管元参数
    matched_env_info = info_raw[seg_ids]
    type_ids = matched_env_info[:,1].long()
    L = matched_env_info[:,3]
    theta = matched_env_info[:,5]
    
    # * 3. 计算目标位姿坐标系中的y轴与z轴方向向量(在管元坐标系下)
    # 类型0: 假设所有输入点匹配到的管元都是直管道
    y_straight = torch.zeros((num_envs , 3), dtype = dtype, device= device)
    y_straight[:,1] = 1.0 # 直管段的匹配点方向向量沿管元坐标系y轴
    z_straight = torch.zeros((num_envs , 3) , dtype= dtype , device= device)
    z_straight[:, 0] = local_xyz[:,0]
    z_straight[:, 2] = local_xyz[:,2]
    
    # 类型1: 假设所有输入点匹配到的管元都是弯管道
    theta_p = axial_ratio * theta
    y_curved = torch.zeros((num_envs , 3) , dtype=dtype , device= device)
    y_curved[:,1] = torch.cos(theta_p)
    y_curved[:,2] = torch.sin(theta_p)
    z_curved = torch.zeros((num_envs , 3) , dtype= dtype , device= device)
    z_curved[: , 0] = local_xyz[:, 0]
    z_curved[:,1] = local_xyz[:, 1] - L * torch.sin(theta_p)
    z_curved[:,2] = local_xyz[:,2] - L * (1.0 - torch.cos(theta_p))
    
    # 按照掩码自动区分管元类型进行融合
    is_straight = (type_ids == 0).unsqueeze(-1)
    y_axis_target = torch.where(is_straight, y_straight, y_curved)
    z_axis_target = torch.where(is_straight, z_straight, z_curved)
    
    # * 4. 向量归一化, 并通过叉乘得到右手系下的x轴
    z_norm = torch.norm(z_axis_target , dim = 1, keepdim= True)
    # 防止出现点精确处在轴心导致的z向量模长为0导致计算爆NaN, 引入一个极小值1e-6
    z_norm = torch.where(z_norm < 1e-6, torch.ones_like(z_norm)*1e-6, z_norm)
    z_axis_target = z_axis_target / z_norm    # 除以模长实现归一化
    
    # 两种求法的y轴都是单位向量, 不需要再进行归一化了, 直接用叉积求x轴
    x_axis_target = torch.cross(y_axis_target, z_axis_target, dim = 1)
    x_norm = torch.norm(x_axis_target, dim = 1, keepdim=True)
    x_norm = torch.where(x_norm < 1e-6 , torch.ones_like(x_norm) * 1e-6, x_norm)
    x_axis_target = x_axis_target / x_norm
    
    # * 5. 组装目标位姿坐标系, 在局部坐标系下的旋转矩阵
    # R ^ pipe _ target  (即: \mathcal{P} 系到 管元局部系 的变换矩阵)
    R_pipe_target = torch.stack([x_axis_target , y_axis_target, z_axis_target], dim = -1)
    
    # * 6. 从输入的矩阵编号中截出完整的管元在世界系下的旋转矩阵块
    trans_env = trans_raw[seg_ids]
    if already_inverted:
        # trans_raw 是 (T^W_pipe)^-1 == T^pipe_W
        R_pipe_W = trans_env[:,0:3,0:3]
    else:
        # trans_raw 是 T^W_pipe，转置后得到旋转的逆 (即 R^pipe_W)
        R_pipe_W = trans_env[:,0:3,0:3].transpose(1, 2)
    
    # * 7. 解析四元数得到机器人在世界坐标系的姿态 R ^ W _ robot
    R_W_robot = pose_utils.quat_wzyz_to_rotmat(quats)
    
    # * 8. 计算 R ^ pipe_robot (机器人在管元系下的位姿)
    R_pipe_robot = R_pipe_W @ R_W_robot
    
    # * 9. 计算在目标位姿坐标系中, 机器人的相对位姿
    # R ^ target_robot = (R^pipe_target)^T @ R^pipe_robot
    # 注意：三维 Tensor 转置必须使用 .transpose(1, 2) 或 .mT，不可直接使用空号 .transpose()
    R_target_robot = R_pipe_target.transpose(1, 2) @ R_pipe_robot
    
    # * 10. 解析最后得到在目标坐标系下的偏差偏角 [num_envs , 2]
    euler_angles = pose_utils.rotmat_to_euler_xyz(R_target_robot)
    theta_x = euler_angles[:, 0]
    theta_z = euler_angles[:, 2]
    
    # 构造成 [N, 2] 结果张量
    angle_diffs = torch.stack([theta_x, theta_z], dim=-1)
    # 对于脱离管道的不合法点（越界），返回角度差为0.0或不做处理
    angle_diffs = torch.where(valid_mask.unsqueeze(1), angle_diffs, torch.zeros_like(angle_diffs))
    return angle_diffs


def is_point_on_pipe(points: torch.Tensor, trans_raw: torch.Tensor, info_raw: torch.Tensor, already_inverted = True) -> torch.Tensor:
    """
    points: [P,3] 世界坐标系下的点坐标
    trans_raw: [N,4,4] 每个管道段的齐次变换矩阵，描述管道段在世界坐标系下的位置和姿态
    info_raw: [N,6] 每个管道段的描述参数
    return: [P,5] 五维向量，包含以下信息：对应点在对应管段的相对坐标xyz, 轴向距离占比，径向距离占比
    return: [x_loacal, y_local , z_local, seg_id + axial_ratio, radial_ratio]
    如果找不到相匹配的管元，则对应的id会被置为-1
    """
    if points.dim() == 1:
        points = points.unsqueeze(0) 

    device = points.device
    dtype = points.dtype
    num_points = points.size(0)
    # 存储每个点在在整个管路上的最好匹配径向距离
    best_radial = torch.full((num_points,), float("inf"), dtype=dtype, device=device)
    best_out = torch.zeros((num_points, 5), dtype=dtype, device=device)

    if already_inverted == False:
        # 如果外部已经预先计算了逆矩阵，则直接使用, 否则在函数内部计算逆矩阵
        trans_raw = torch.linalg.inv(trans_raw)
    for i in range(info_raw.size(0)):
        seg_out = is_point_on_segment(points, trans_raw[i], info_raw[i, :])
        axial = seg_out[:, 3]
        radial = seg_out[:, 4]
        valid = (axial >= 0.0) & (axial <= 1.0) & (radial >= 1.0) # 结果合法性
        better = valid & (radial < best_radial)
        if better.any(): # 只要better有数据
            seg_out = seg_out.clone()
            seg_out[:, 3] = seg_out[:, 3] + i
            best_radial = torch.where(better, radial, best_radial)
            best_out = torch.where(better.unsqueeze(1), seg_out, best_out)

    no_hit = torch.isinf(best_radial)
    if no_hit.any():
        best_out = torch.where(no_hit.unsqueeze(1), torch.zeros_like(best_out), best_out)
        best_out[no_hit, 3] = -1.0
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
    # 选择测试计算设备 (自动优先GPU)
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_device = torch.device("cpu")  # 强制使用CPU进行测试，便于调试和验证结果
    print(f"[INFO] Test device: {test_device}")
    # * 读取并挂载本轮随机加载的JSON管道描述文件
    if os.path.exists(SELECTED_PIPE_JSON):
        with open(SELECTED_PIPE_JSON, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        transform_list = [item["transform"] for item in raw_data]
        info_list = [item["info"] for item in raw_data]
        
        # 直接转换为Tensor， PyTorch会自动推断并形成对应形状
        pipe_transform = torch.tensor(transform_list, dtype=torch.float32, device=test_device)  # [N, 4, 4]
        pipe_info = torch.tensor(info_list, dtype=torch.float32, device=test_device)            # [N, 6]
        print(f"[INFO] Pipe Transform Tensor Shape: {pipe_transform.shape}")
        print(f"[INFO] Pipe Info Tensor Shape: {pipe_info.shape}")
    else:
        # Shape: [1, 4, 4]
        pipe_transform = torch.eye(4, device=test_device).unsqueeze(0)
        # Shape: [1, 6]
        pipe_info = torch.tensor([[0, 0, 0.36, 1.0, 0.0, 0.0]], dtype=torch.float32, device=test_device)
        print(f"[WARNING] Could not find JSON info at: {SELECTED_PIPE_JSON}")
    
    # 使用单个管道测试相对坐标转换 (批量形式)
    seg_id = 0
    test_points = torch.tensor(
        [
            [0.0, 1.2 -1.5, 0.5],
            [0.0, 1.2 + 1.5, 0.5],
            [0.2, 0.4, 0.1],
            [-0.3, 0.8, 0.2],
        ],
        dtype=torch.float32,
        device=test_device,
    )
    rand_points = torch.rand((1024, 3), dtype=torch.float32, device=test_device) * 2.0 - 1.0  # [-1, 1]
    test_points = torch.cat([test_points, rand_points], dim=0)
    # 使用整段管道完成主函数测试
    pipe_transform_inv = torch.linalg.inv(pipe_transform) # 预先计算逆矩阵，减少后续计算量
    if test_device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = is_point_on_pipe(test_points, pipe_transform_inv, pipe_info)
    if test_device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"[PERF] is_point_on_pipe elapsed: {elapsed_ms:.3f} ms")
    print("\n================= Query Result =================")
    # result = [x_local, y_local, z_local, (seg_id + axis_ratio), radial_ratio]
    for idx in range(test_points.size(0)):
        if idx >= 5:  # 只打印前5个点的结果，避免过长输出
            print(f"... (skipping remaining {test_points.size(0) - 5} points) ...")
            break
        point = test_points[idx]
        out = result[idx]
        print(f"world point            : {point.tolist()}")
        if out[3].item() < 0.0:
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


