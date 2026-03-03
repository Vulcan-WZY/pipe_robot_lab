# ===========
# Date: 2026-03-02 10:48
# Author: Vulcan
# LastEditTime: 2026-03-02 11:37
# Description: 用来存放一些处理四元数，欧拉角以及旋转矩阵的工具函数
# ==========
import torch

# 四元数转换到旋转矩阵
def quat_wzyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: [N,4] (w,x,y,z) -> R: [N,3,3]"""
    w, x ,y, z = q[: , 0] , q[:, 1] , q[:,2], q[:,3]
    ww, xx , yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    R = torch.zeros((q.shape[0], 3, 3), dtype= q.dtype, device= q.device)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R

# 旋转矩阵转换到欧拉角
def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """R: [N,3,3] -> euler_xyz: [N,3]"""
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6

    roll = torch.where(
        ~singular,
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
        torch.atan2(-R[:, 1, 2], R[:, 1, 1]),
    )
    pitch = torch.where(
        ~singular,
        torch.atan2(-R[:, 2, 0], sy),
        torch.atan2(-R[:, 2, 0], sy),
    )
    yaw = torch.where(
        ~singular,
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        torch.zeros_like(roll),
    )
    return torch.stack([roll, pitch, yaw], dim=-1)

