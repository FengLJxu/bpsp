import open3d as o3d
import numpy as np
import time


def load_and_prepare_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename, format='xyzrgb')
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()
    pcd.normalize_normals()
    return pcd

def segment_by_color(pcd):
    colors = np.asarray(pcd.colors)
    red_color = np.array([255, 0, 0])  # 浮点数表示的红色
    blue_color = np.array([0, 0, 255])  # 浮点数表示的蓝色
    plane_indices = np.where((colors[:, 0] == red_color[0]) & (colors[:, 1] == red_color[1]) & (colors[:, 2] == red_color[2]))[0]
    non_plane_indices = np.where((colors[:, 0] == blue_color[0]) & (colors[:, 1] == blue_color[1]) & (colors[:, 2] == blue_color[2]))[0]
    return plane_indices, non_plane_indices

def compute_geometric_properties(pcd, indices):
    selected_pcd = pcd.select_by_index(indices)
    curvature = np.abs(np.asarray(selected_pcd.normals)[:, 2])
    return curvature, np.asarray(selected_pcd.normals)



# 定义区域生长分割函数
def region_growing_segmentation(pcd, kdtree, distance_threshold, angle_threshold, min_points):
    curvatures, normals = compute_geometric_properties(pcd, np.arange(len(pcd.points)))
    indices = np.argsort(curvatures)
    segments = []
    visited = np.zeros(len(pcd.points), dtype=bool)
    for i in indices:
        if visited[i]:
            continue
        seeds = [i]
        segment = [i]
        visited[i] = True
        while seeds:
            seed = seeds.pop(0)
            seed_normal = normals[seed]
            _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[seed], 30)
            for j in idx[1:]:
                if visited[j]:
                    continue
                neighbor_normal = normals[j]
                dot_product = np.clip(np.dot(seed_normal, neighbor_normal), -1.0, 1.0)
                angle = np.arccos(dot_product)
                if angle * 180 / np.pi < angle_threshold:
                    distance = np.linalg.norm(pcd.points[seed] - pcd.points[j])
                    if distance < distance_threshold:
                        seeds.append(j)
                        segment.append(j)
                        visited[j] = True
        if len(segment) >= min_points:
            segments.append(segment)
    return segments

def merge_non_plane_to_plane(plane_pcd, non_plane_pcd, plane_curvatures, plane_normals, non_plane_curvature, non_plane_normals):
    # 初始化最大曲率差和最大角度差
    max_curvature_diff = 0
    max_angle = 0
    max_distance = 0

    # 计算最大曲率差和最大角度差
    for nc in non_plane_curvature:
        max_curvature_diff = max(max_curvature_diff, np.max(np.abs(plane_curvatures - nc)))
    for n_norm in non_plane_normals:
        angles = np.arccos(np.clip(np.dot(plane_normals, n_norm), -1.0, 1.0))
        max_angle = max(max_angle, np.max(angles))

    # 计算最大距离
    for p_point in non_plane_pcd.points:
        distances = np.linalg.norm(np.asarray(plane_pcd.points) - p_point, axis=1)
        max_distance = max(max_distance, np.max(distances))

    non_plane_points = np.asarray(non_plane_pcd.points)
    plane_points = np.asarray(plane_pcd.points)
    plane_kdtree = o3d.geometry.KDTreeFlann(plane_pcd)

    for i, point in enumerate(non_plane_points):
        min_score = float('inf')
        best_plane_idx = None
        [k, idx, _] = plane_kdtree.search_knn_vector_3d(point, 10)
        for j in idx:
            plane_point = plane_points[j]
            curvature_diff = abs(non_plane_curvature[i] - plane_curvatures[j])
            angle_diff = np.arccos(np.clip(np.dot(non_plane_normals[i], plane_normals[j]), -1.0, 1.0))
            distance = np.linalg.norm(point - plane_point)
            # 增加分数计算的复杂性和准确性
            score = (curvature_diff / max_curvature_diff + angle_diff / max_angle + distance / max_distance) / 3  # 使用平均分数进行评估

            if score < min_score:
                min_score = score
                best_plane_idx = j
        if best_plane_idx is not None:
            non_plane_pcd.colors[i] = plane_pcd.colors[best_plane_idx]

    merged_pcd = plane_pcd + non_plane_pcd
    return merged_pcd


def merge_segments(pcd, segments):
    colors = np.zeros((len(pcd.points), 3))  # 初始化颜色数组
    for segment in segments:
        color = np.random.rand(3)  # 为每个分割生成一个随机颜色
        for idx in segment:
            colors[idx] = color  # 应用颜色到所有分割中的点上
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 更新点云的颜色属性
    return pcd


def main():
    start_time = time.time()
    pcd = load_and_prepare_point_cloud("combined1_cathedral1(7 48).txt")
    plane_indices, non_plane_indices = segment_by_color(pcd)
    plane_pcd = pcd.select_by_index(plane_indices, invert=False)
    non_plane_pcd = pcd.select_by_index(non_plane_indices, invert=False)
    plane_kdtree = o3d.geometry.KDTreeFlann(plane_pcd)
    non_plane_kdtree = o3d.geometry.KDTreeFlann(non_plane_pcd)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_plane = executor.submit(region_growing_segmentation, plane_pcd, plane_kdtree, 0.3, 4, 5)
        future_non_plane = executor.submit(region_growing_segmentation, non_plane_pcd, non_plane_kdtree, 0.7, 5, 5)

    segments_plane = future_plane.result()
    segments_non_plane = future_non_plane.result()
    merged_plane_pcd = merge_segments(plane_pcd, segments_plane)
    merged_non_plane_pcd = merge_segments(non_plane_pcd, segments_non_plane)
    plane_curvatures, plane_normals = compute_geometric_properties(merged_plane_pcd,
                                                                   np.arange(len(merged_plane_pcd.points)))
    non_plane_curvature, non_plane_normals = compute_geometric_properties(merged_non_plane_pcd,
                                                                          np.arange(len(merged_non_plane_pcd.points)))
    final_merged_pcd = merge_non_plane_to_plane(merged_plane_pcd, merged_non_plane_pcd, plane_curvatures, plane_normals,
                                                non_plane_curvature, non_plane_normals)

    # 检查未分配点并尝试重新分配
    visited = np.array([False] * len(pcd.points))
    for segment in segments_plane + segments_non_plane:
        visited[segment] = True
    unassigned_points = np.where(visited == False)[0]

    # 为未分配的点寻找最近的分段并分配颜色
    for point_idx in unassigned_points:
        _, idx, _ = plane_kdtree.search_knn_vector_3d(pcd.points[point_idx], 15)
        nearest_segment_idx = [seg for seg in segments_plane + segments_non_plane if idx[0] in seg]
        if nearest_segment_idx:
            pcd.colors[point_idx] = pcd.colors[nearest_segment_idx[0][0]]

    # 可视化并保存结果
    # o3d.visualization.draw_geometries([final_merged_pcd])
    o3d.io.write_point_cloud("final_result_cathedral10.pcd", final_merged_pcd, write_ascii=True, print_progress=True)
    end_time = time.time()
    print(f"运行时间: {end_time - start_time} 秒")


if __name__ == "__main__":
    main()
