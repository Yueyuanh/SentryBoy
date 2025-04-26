import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(grid, start, end):
    open_list = []
    closed_set = set()
    
    start_node = Node(start)
    end_node = Node(end)
    
    heapq.heappush(open_list, start_node)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for dx, dy in directions:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            if (0 <= neighbor_pos[0] < grid.shape[0] and
                0 <= neighbor_pos[1] < grid.shape[1] and
                grid[neighbor_pos[0], neighbor_pos[1]] == 255 and
                neighbor_pos not in closed_set):

                move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1

                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + move_cost
                neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                if any(open_node.position == neighbor_node.position and open_node.f <= neighbor_node.f for open_node in open_list):
                    continue

                heapq.heappush(open_list, neighbor_node)

    return None

def jps(grid, start, end):
    from collections import deque
    import math

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),  # 四方向
                  (1, 1), (1, -1), (-1, 1), (-1, -1)] # 八方向

    def forced_neighbors(pos, parent_pos):
        # 实现跳跃点检测逻辑（此处简化）
        # 完整实现需处理斜向跳跃和强迫邻居
        x, y = pos
        px, py = parent_pos if parent_pos else (x, y)
        dx, dy = x - px, y - py
        
        # 简化版：直接返回所有可行方向
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 255:
                neighbors.append((nx, ny))
        return neighbors

    open_set = {start: 0}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    open_heap = [(f_score[start], start)]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in forced_neighbors(current, came_from.get(current)):
            tentative_g = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                    open_set[neighbor] = 1

    return None

def inflate_obstacles(binary_map, radius):
    """对障碍物进行膨胀处理，同时返回膨胀区域mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    inflated = cv2.dilate(255 - binary_map, kernel)
    result = 255 - inflated

    # 计算膨胀区域mask（1表示膨胀新增的区域）
    inflate_mask = np.logical_and(result == 0, binary_map == 255).astype(np.uint8)
    return result, inflate_mask

def plot_paths(original_map, inflated_map, inflate_mask, path, start, end):
    """专业风格可视化地图 - 输出两张图像"""
    # 第一张图：膨胀后的地图加路径
    img_inflated = np.full((*inflated_map.shape, 3), 255, dtype=np.uint8)  # 全白底
    
    # 原始障碍物（深灰）
    img_inflated[original_map == 0] = (50, 50, 50)
    
    # 膨胀区域（蓝）
    img_inflated[inflate_mask == 1] = (255, 0, 0)
    
    # 路径轨迹（橘红，线宽加粗）
    for y, x in path:
        img_inflated[y, x] = (0, 100, 255)
    
    # 起点（青绿色，增大标记）
    img_inflated[start[0], start[1]] = (0, 255, 255)
    
    # 终点（品红）
    img_inflated[end[0], end[1]] = (255, 0, 255)

    # 第二张图：原始地图上的路径（无膨胀可视化）
    img_original = np.full((*original_map.shape, 3), 255, dtype=np.uint8)  # 全白底
    
    # 原始障碍物（深灰）
    img_original[original_map == 0] = (50, 50, 50)
    
    # 路径轨迹（橘红，线宽加粗）
    for y, x in path:
        img_original[y, x] = (0, 100, 255)
    
    # 起点（青绿色，增大标记）
    img_original[start[0], start[1]] = (0, 255, 255)
    
    # 终点（品红）
    img_original[end[0], end[1]] = (255, 0, 255)

    # 显示两张图
    plt.figure(figsize=(20, 10))
    
    # 第一张图：带膨胀障碍物的路径
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_inflated, cv2.COLOR_BGR2RGB))
    plt.title("A* Path with Inflated Obstacles")
    plt.axis('off')
    
    # 第二张图：原始地图上的路径
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("A* Path on Original Map")
    plt.axis('off')
    
    # 加粗路径（增加linewidth）
    plt.subplot(1, 2, 1).plot([x for _, x in path], [y for y, _ in path], '-o', color='green', markersize=2, linewidth=1, label='Path')
    plt.subplot(1, 2, 2).plot([x for _, x in path], [y for y, _ in path], '-o', color='green', markersize=2, linewidth=1, label='Path')

    # 加粗起始点（增加marker大小）
    plt.subplot(1, 2, 1).scatter(start[1], start[0], c='cyan', s=500, edgecolors='black', label='Start', marker='o')
    plt.subplot(1, 2, 2).scatter(start[1], start[0], c='cyan', s=500, edgecolors='black', label='Start', marker='o')
    
    # 加粗终点（增加marker大小）
    plt.subplot(1, 2, 1).scatter(end[1], end[0], c='magenta', s=500, edgecolors='black', label='End', marker='s')
    plt.subplot(1, 2, 2).scatter(end[1], end[0], c='magenta', s=500, edgecolors='black', label='End', marker='s')

    # 显示图例
    plt.tight_layout()
    plt.savefig("path_planning_result.png")
    plt.show()

# ---------------- 主程序 ----------------

# 配置
img_path = "map/RMUC1_B.png"  # 地图路径
inflate_radius = 10              # 膨胀半径

# 读取图
gray_map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 二值化
_, binary_map = cv2.threshold(gray_map, 127, 255, cv2.THRESH_BINARY)

h, w = binary_map.shape

# 起终点
# start = (h - 1, 0)   # 左下角
# end = (0, w - 1)     # 右上角
start = (220, 120)         
end   = (220, 680)   

# 膨胀处理
inflated_map, inflate_mask = inflate_obstacles(binary_map, inflate_radius)

# 计时
start_time = time.time()

# 路径规划
# path = astar(inflated_map, start, end)
# 使用JPS算法进行路径规划
path = jps(inflated_map, start, end)

end_time = time.time()
elapsed_time = end_time - start_time

# 输出
if path:
    print(f"路径长度: {len(path)}")
    print(f"规划耗时: {elapsed_time:.4f} 秒")
    plot_paths(binary_map, inflated_map, inflate_mask, path, start, end)
else:
    print("未找到路径。")