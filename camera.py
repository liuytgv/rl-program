import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from assit import assit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def update_camera_view(ax, d, aspect_ratio, camera_pos, azimuth, elevation, focal_length, R1, R2, R3, dtc, oi):

    # 清除坐标系中的所有几何面
    ax.collections.clear
    
    # 清除相机视场图像
    lines = ax.get_lines()
    for line in lines:
        if line.get_color() in ['lightblue']:
            ax.lines.remove(line)
    ax.clear()
    azimuth_g = np.deg2rad(azimuth)  # 将角度转换为弧度
    elevation_g = np.deg2rad(elevation)  # 将角度转换为弧度

    # 计算相机观察方向
    direction = np.array([np.cos(azimuth_g), np.sin(azimuth_g), np.tan(elevation_g)])
    # print(direction)

    # 绘制相机位置
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='red', marker='o')
    ax.text(camera_pos[0], camera_pos[1], camera_pos[2], 'camera', color='black')

    # 绘制相机朝向向量
    direction_length = 1000  # 长度因子
    direction_end = camera_pos + direction * direction_length
    ax.plot([camera_pos[0], direction_end[0]], [camera_pos[1], direction_end[1]], [camera_pos[2], direction_end[2]],
        color='lightblue', alpha=0.5)


    # 相机的水平视场
    # print(d, focal_length)
    fov_adjusted = np.rad2deg(2 * np.arctan(0.5 * d / focal_length))
    # print(fov_adjusted)

    # 相机的上下俯仰视场
    fov_adjusted1 = fov_adjusted / aspect_ratio

    # 第一条
    azimuth_a = np.deg2rad(azimuth + fov_adjusted / 2)  # 将角度转换为弧度
    elevation_a = np.deg2rad(elevation + fov_adjusted1 / 2)  # 将角度转换为弧度
    direction_a = np.array([np.cos(azimuth_a), np.sin(azimuth_a), np.tan(elevation_a)])
    d_r1 = camera_pos + direction_a * R1
    d_rr1 = camera_pos + direction_a * R2
    d_rrr1 = camera_pos + direction_a * R3


    # 第二条
    azimuth_b = np.deg2rad(azimuth + fov_adjusted / 2)  # 将角度转换为弧度
    elevation_b = np.deg2rad(elevation - fov_adjusted1 / 2)  # 将角度转换为弧度
    direction_b = np.array([np.cos(azimuth_b), np.sin(azimuth_b), np.tan(elevation_b)])
    d_r2 = camera_pos + direction_b * R1
    d_rr2 = camera_pos + direction_b * R2
    d_rrr2 = camera_pos + direction_b * R3

    # 第三条
    azimuth_c = np.deg2rad(azimuth - fov_adjusted / 2)  # 将角度转换为弧度
    elevation_c = np.deg2rad(elevation + fov_adjusted1 / 2)  # 将角度转换为弧度
    direction_c = np.array([np.cos(azimuth_c), np.sin(azimuth_c), np.tan(elevation_c)])
    d_r3 = camera_pos + direction_c * R1
    d_rr3 = camera_pos + direction_c * R2
    d_rrr3 = camera_pos + direction_c * R3

    # 第四条 
    azimuth_d = np.deg2rad(azimuth - fov_adjusted / 2)  # 将角度转换为弧度
    elevation_d = np.deg2rad(elevation - fov_adjusted1 / 2)  # 将角度转换为弧度
    direction_d = np.array([np.cos(azimuth_d), np.sin(azimuth_d), np.tan(elevation_d)])
    d_r4 = camera_pos + direction_d * R1
    d_rr4 = camera_pos + direction_d * R2
    d_rrr4 = camera_pos + direction_d * R3

    # 绘制几何面
    vertices = [
        [d_r1, d_r2, camera_pos],
        [d_r2, d_r4, camera_pos],
        [d_r4, d_r3, camera_pos],
        [d_r3, d_r1, camera_pos]
    ]

    poly3d = [[vertices[i][j] for j in range(3)] for i in range(4)]

    if dtc <= R1 and oi == 1:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='lightcoral', linewidths=1, edgecolors='red', alpha=.25))

    else:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='red', linewidths=1, edgecolors='red', alpha=.40))

    vertices = [
        [d_r1, d_rr1, d_rr2, d_r2],
        [d_r2, d_rr2, d_rr4, d_r4],
        [d_r4, d_rr4, d_rr3, d_r3],
        [d_r3, d_rr3, d_rr1, d_r1],
        [d_rr3, d_rr4, d_rr2, d_rr1]
    ]

    poly3d = [[vertices[i][j] for j in range(4)] for i in range(5)]

    if R1 <= dtc <= R2 and oi == 1:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='yellow', linewidths=1, edgecolors='orange', alpha=.25))
    
    else:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='orange', linewidths=1, edgecolors='orange', alpha=.40))
    
    vertices = [
        [d_rr1, d_rrr1, d_rrr2, d_rr2],
        [d_rr2, d_rrr2, d_rrr4, d_rr4],
        [d_rr4, d_rrr4, d_rrr3, d_rr3],
        [d_rr3, d_rrr3, d_rrr1, d_rr1],
        [d_rrr3, d_rrr4, d_rrr2, d_rrr1]
    ]

    poly3d = [[vertices[i][j] for j in range(4)] for i in range(5)]

    if R2 <= dtc <= R3 and oi == 1:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='lightgreen', linewidths=1, edgecolors='green', alpha=.25))
    
    else:
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='green', linewidths=1, edgecolors='green', alpha=.40))

# 调用函数来绘制三个几何体
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.set_zlim([0, 100])
# camera_pos = np.array([0, 0, 0])
# d = 0.00818
# aspect_ratio = 16/9
# azimuth = 0
# elevation = 60
# focal_length = 0.09
# R1, R2, R3 = assit(focal_length)
# update_camera_view(ax, d, aspect_ratio, camera_pos, azimuth, elevation, focal_length, R1, R2, R3)
# plt.show()
# plt.close(fig)