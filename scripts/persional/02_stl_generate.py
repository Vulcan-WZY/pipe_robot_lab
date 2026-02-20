import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

def generate_bent_pipe(diameter, radius, angle_degrees, num_step=5, num_points=35):
    """
    生成弯管的STL文件
    :param diameter: 弯管直径
    :param radius: 弯曲半径（中心线半径）
    :param angle_degrees: 弯曲角度（圆心角，度）
    :param num_step: 中心线的角度步长数（沿管长方向的分段数）
    :param num_points: 横截面的点数
    """
    # 转换角度为弧度
    angle = np.radians(angle_degrees)
    num_segments = angle_degrees // ( num_step) + 1  # 根据角度和步长计算分段数
    
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
    
    return pipe_mesh

# 生成弯管
bent_pipe = generate_bent_pipe(diameter=0.36, radius=1.3, angle_degrees=45)

# 保存为STL文件
bent_pipe.save('bent_pipe.stl')

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 修复报错：使用 art3d.Poly3DCollection 加载 mesh 的 vectors
poly_collection = art3d.Poly3DCollection(bent_pipe.vectors)
poly_collection.set_edgecolor('k')  # 设置网格线颜色
poly_collection.set_alpha(0.6)      # 设置透明度
ax.add_collection3d(poly_collection)

# 设置坐标轴显示范围，防止图形被压缩
max_range = np.array([bent_pipe.x.max()-bent_pipe.x.min(), 
                    bent_pipe.y.max()-bent_pipe.y.min(), 
                    bent_pipe.z.max()-bent_pipe.z.min()]).max() / 2.0

mid_x = (bent_pipe.x.max()+bent_pipe.x.min()) * 0.5
mid_y = (bent_pipe.y.max()+bent_pipe.y.min()) * 0.5
mid_z = (bent_pipe.z.max()+bent_pipe.z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()