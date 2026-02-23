import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

# 定义你的原始 STL 文件路径
source_path = "source/pipe_robot_lab/pipe_robot_lab/assets/pipe_env/meshes/stand_pipe.STL"



if os.path.exists(source_path):
    print(f"正在加载: {source_path}")
    
    # 1. 加载网格
    try:
        my_mesh = mesh.Mesh.from_file(source_path)
        print("网格加载成功")
    except Exception as e:
        print(f"加载失败: {e}")
        exit()

    # --- 新增：平移修正代码 ---
    # 计算当前包围盒
    all_points = my_mesh.points.reshape(-1, 3)
    min_xyz = np.min(all_points, axis=0)
    max_xyz = np.max(all_points, axis=0)
    
    # 假设管径为1m (R=0.5m)，且长度沿Y轴
    # 用户指出当前端面圆心在 (R, 0, R)
    # 实际上可以通过计算 X 和 Z 的中点来验证和修正
    center_x = (min_xyz[0] + max_xyz[0]) / 2.0
    center_z = (min_xyz[2] + max_xyz[2]) / 2.0
    
    # 构建平移变换矩阵：将中心移回 (0, 0, 0)
    # 只需要在 X 和 Z 方向移动，Y 方向保持从 0 开始 (min_y)
    translation_vector = [-center_x, 0.0, -center_z] # 保持 Y 轴底面不动，如果原点是 Y=0
    # 如果要将 Z 也归零（假设用户说的R是半径偏移）
    
    print(f"检测到中心偏差: X={center_x}, Z={center_z}")
    print(f"执行平移: {translation_vector}")
    
    my_mesh.translate(translation_vector)

    # 保存修改后的网格
    my_mesh.save(source_path)
    print(f"文件已更新并保存至: {source_path}")
    # -----------------------

    # 2. 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制坐标轴 (Origin at 0,0,0)
    origin = [0, 0, 0]
    length = 1.0 # 坐标轴长度，单位米
    
    # X轴 - 红色
    ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='World X')
    # Y轴 - 绿色 
    ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='World Y')
    # Z轴 - 蓝色
    ax.quiver(0, 0, 0, 0, 0, length, color='b', arrow_length_ratio=0.1, linewidth=2, label='World Z')

    # 将 STL 网格添加到图形中
    # mpl_toolkits 需要 Poly3DCollection
    if my_mesh.vectors.size > 0:
        poly_collection = art3d.Poly3DCollection(my_mesh.vectors)
        poly_collection.set_edgecolor('k') # 黑色边框
        poly_collection.set_alpha(0.5)     # 半透明
        # poly_collection.set_facecolor('cyan') 
        ax.add_collection3d(poly_collection)

        # 计算包围盒来设置坐标轴范围
        # my_mesh.points 是 (N, 9) 的数组 (3个点 * 3个坐标)
        all_points = my_mesh.points.reshape(-1, 3)
        min_xyz = np.min(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)
        
        print(f"包围盒范围:\nMin (x,y,z): {min_xyz}\nMax (x,y,z): {max_xyz}")
        
        # 自动调整视角范围
        max_range = np.array([max_xyz[0]-min_xyz[0], max_xyz[1]-min_xyz[1], max_xyz[2]-min_xyz[2]]).max() / 2.0
        mid_x = (max_xyz[0]+min_xyz[0]) * 0.5
        mid_y = (max_xyz[1]+min_xyz[1]) * 0.5
        mid_z = (max_xyz[2]+min_xyz[2]) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        print("网格为空！")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 添加图例
    # 由于 quiver 不直接支持图例，我们可以创建代理 artists
    # 或者直接 plt.legend() 有时也能捕捉到
    ax.text(length, 0, 0, "X", color='red')
    ax.text(0, length, 0, "Y", color='green')
    ax.text(0, 0, length, "Z", color='blue')

    plt.show()

else:
    print(f"错误：找不到文件 {source_path}")