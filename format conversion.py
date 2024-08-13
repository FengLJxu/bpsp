import open3d as o3d
import numpy as np

# 读取点云文件
pcd = o3d.io.read_point_cloud("final_result_cathedral4.pcd")

# 将点云数据转换为numpy数组
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 将颜色从浮点数转换为0-255范围内的整数
colors_int = (colors * 255).astype(np.int)

# 组合xyz和rgb数据
xyzrgb = np.hstack((points, colors_int))

# 保存为txt文件
np.savetxt("final_result_cathedral4.txt", xyzrgb, fmt='%f %f %f %d %d %d')

print("保存完成。")
