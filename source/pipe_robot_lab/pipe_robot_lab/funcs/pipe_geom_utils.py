# ===========
# Date: 2026-03-02 11:41
# Author: Vulcan
# LastEditTime: 2026-03-02 16:30
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
    # info: 管道编号， 管道类型， 管道直径， 管道长度， 前置偏转， 后置偏转    
    type = info[1]
    if type == 0: # 直管段
        pass
    elif type == 1: # 弯管
        pass



