import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import time

start_time = time.time()
# 读取点云数据
pcd = o3d.io.read_point_cloud("cathedra2 out .txt", format='xyzrgb')


# 获取点云颜色和位置
colors = np.asarray(pcd.colors)
points = np.asarray(pcd.points)

# 构建颜色标签字典
color_clusters = defaultdict(list)
for i, color in enumerate(colors):
    label = tuple(color)
    color_clusters[label].append(i)

# 合并相同颜色的点云
clustered_points = defaultdict(list)
for label, indices in color_clusters.items():
    if len(indices) < 2:
        continue
    for index in indices:
        clustered_points[label].append(points[index])

# 判断平面和非平面超体素
plane_clusters = []
non_plane_clusters = []

for color_label, cluster_points in clustered_points.items():
    if len(cluster_points) < 3:
        continue
    points_array = np.array(cluster_points)

    # 自适应地选择邻居数量
    n_neighbors = min(6, len(cluster_points) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points_array)
    distances, _ = nbrs.kneighbors(points_array)
    avg_distance = distances.mean()

    # 自适应阈值选择
    distance_threshold = np.percentile(distances, 42)  # 使用中位数距离作为阈值
    if avg_distance < distance_threshold:
        plane_clusters.append(cluster_points)
    else:
        non_plane_clusters.append(cluster_points)

# 合并结果并设置颜色
plane_cloud = o3d.geometry.PointCloud()
non_plane_cloud = o3d.geometry.PointCloud()
for points in plane_clusters:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    plane_cloud += cloud
for points in non_plane_clusters:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    non_plane_cloud += cloud

plane_cloud.paint_uniform_color([1, 0, 0])  # 平面超体素用红色表示
non_plane_cloud.paint_uniform_color([0, 0, 1])  # 非平面超体素用蓝色表示

# 可视化结果
o3d.visualization.draw_geometries([plane_cloud, non_plane_cloud], width=1200, height=600)

# 保存结果
o3d.io.write_point_cloud("cathedral1.(6 42) p.pcd", plane_cloud)
o3d.io.write_point_cloud("cathedral1.(6 42) np.pcd", non_plane_cloud)
end_time = time.time()
print(f"运行时间: {end_time - start_time} 秒")