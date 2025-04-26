import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from datetime import datetime
import os
import csv
from scipy.interpolate import splprep, splev
import math

# 全局配置
CONFIG = {
    "map_path": "map/RMUC1_B.png",
    "robot_path": "robot/sentry1.png",  # 机器人图片路径
    "inflate_radius": 10,
    "output_dir": "results",
    "default_speed": 1,  # 默认移动速度(ms/点)
    "animation_scale": 2.0,  # 动画窗口放大倍数
    "smoothing_factor": 0.5,  # 路径平滑因子(0-1)，越大越平滑
    "spline_points": 200,   # 样条曲线插值点数
    "front_box_length": 30,  # 前方矩形框长度(像素)
    "front_box_width": 25,   # 前方矩形框宽度(像素)
    "box_color": (255, 165, 0),  # 矩形框颜色(橙色)
    "intersection_color": (255, 0, 255),  # 交点颜色(紫色)
    "intersection_size": 10,  # 交点标记大小

    "save_animation": True,
    "animation_video_path": "robot_animation.mp4",
    "animation_fps": 15,
}

class PathPlanner:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.click_count = 0
        self.binary_map = None
        self.inflated_map = None
        self.inflate_mask = None
        self.running = True
        self.robot_img = None
        self.path = None
        self.smooth_path = None
        self.animation_speed = CONFIG["default_speed"]
        self.front_box_pts = None  # 前方矩形框的四个顶点
        self.intersection_points = []  # 矩形框与路径的交点

    def load_map(self):
        """加载并预处理地图"""
        gray_map = cv2.imread(CONFIG["map_path"], cv2.IMREAD_GRAYSCALE)
        if gray_map is None:
            raise FileNotFoundError(f"无法加载地图 {CONFIG['map_path']}")
        
        _, self.binary_map = cv2.threshold(gray_map, 127, 255, cv2.THRESH_BINARY)
        self.inflated_map, self.inflate_mask = self.inflate_obstacles(
            self.binary_map, CONFIG["inflate_radius"])
        
        # 加载并调整机器人图片大小
        self.robot_img = cv2.imread(CONFIG["robot_path"], cv2.IMREAD_UNCHANGED)
        if self.robot_img is None:
            print(f"警告: 无法加载机器人图片 {CONFIG['robot_path']}, 将使用默认标记")
            self.robot_img = np.zeros((60, 60, 4), dtype=np.uint8)  # 更大的默认标记
            cv2.circle(self.robot_img, (30, 30), 25, (0, 255, 0, 255), -1)
        else:
            # 放大机器人图片
            new_size = (int(self.robot_img.shape[1] * CONFIG["animation_scale"]), 
                       int(self.robot_img.shape[0] * CONFIG["animation_scale"]))
            self.robot_img = cv2.resize(self.robot_img, new_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def inflate_obstacles(binary_map, radius):
        """障碍物膨胀处理"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        inflated = cv2.dilate(255 - binary_map, kernel)
        result = 255 - inflated
        inflate_mask = np.logical_and(result == 0, binary_map == 255).astype(np.uint8)
        return result, inflate_mask

    def on_mouse_click(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
                self.start_point = (y, x)  # 存储为(y,x)格式
                print(f"起点设置: (y={y}, x={x})")
                self.click_count += 1
            elif self.click_count == 1:
                self.end_point = (y, x)
                print(f"终点设置: (y={y}, x={x})")
                self.click_count += 1
                # 先显示终点，再延迟执行规划
                cv2.circle(self.display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Path Planning - Click to set start/end points", self.display_img)
                cv2.waitKey(100)  # 短暂延迟确保显示更新
                self.run_path_planning()

    def astar(self, grid, start, end):
        """A*路径规划算法"""
        open_list = []
        closed_set = set()
        
        start_node = Node(start)
        end_node = Node(end)
        heapq.heappush(open_list, start_node)

        # 8方向移动
        directions = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]

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
                neighbor_pos = (current_node.position[0]+dx, current_node.position[1]+dy)

                if (0 <= neighbor_pos[0] < grid.shape[0] and
                    0 <= neighbor_pos[1] < grid.shape[1] and
                    grid[neighbor_pos[0], neighbor_pos[1]] == 255 and
                    neighbor_pos not in closed_set):

                    move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1
                    neighbor_node = Node(neighbor_pos, current_node)
                    neighbor_node.g = current_node.g + move_cost
                    neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    if any(open_node.position == neighbor_node.position and 
                          open_node.f <= neighbor_node.f for open_node in open_list):
                        continue

                    heapq.heappush(open_list, neighbor_node)
        return None
    
    def jps(self, grid, start, end):
        """Jump Point Search算法"""
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
    
    def smooth_path_spline(self, path, s=0.0, k=3):
        """使用样条曲线平滑路径"""
        if len(path) < 4:
            return path  # 点数太少，无法平滑
            
        # 提取坐标
        y_coords, x_coords = zip(*path)
        
        # 创建参数化的样条曲线
        tck, u = splprep([y_coords, x_coords], s=s, k=k)
        
        # 生成平滑路径点
        new_points = np.linspace(0, 1, CONFIG["spline_points"])
        smooth_y, smooth_x = splev(new_points, tck)
        
        # 将平滑路径转换为整数坐标
        smooth_path = [(int(y), int(x)) for y, x in zip(smooth_y, smooth_x)]
        
        # 确保起点和终点不变
        smooth_path[0] = path[0]
        smooth_path[-1] = path[-1]
        
        # 检查平滑路径是否穿过障碍物
        valid_path = []
        for point in smooth_path:
            y, x = point
            # 确保点在地图范围内
            if 0 <= y < self.inflated_map.shape[0] and 0 <= x < self.inflated_map.shape[1]:
                # 如果点在障碍物上，跳过
                if self.inflated_map[y, x] == 0:
                    continue
            else:
                continue
            valid_path.append(point)
            
        # 如果有效路径为空，返回原始路径
        if not valid_path:
            return path
            
        return valid_path

    def calculate_slope(self, point1, point2):
        """计算两点之间的斜率"""
        y1, x1 = point1
        y2, x2 = point2
        
        if x2 == x1:  # 避免除以零
            return float('inf') if y2 > y1 else float('-inf')
            
        return (y2 - y1) / (x2 - x1)
    
    def calculate_front_box(self, position, angle):
        """计算机器人前方的矩形框顶点"""
        y, x = position
        angle_rad = math.radians(angle)
        
        # 矩形框尺寸
        length = CONFIG["front_box_length"]
        width = CONFIG["front_box_width"]
        
        # 中心点
        center_x = x + length/2 * math.cos(angle_rad)
        center_y = y + length/2 * math.sin(angle_rad)
        
        # 计算四个顶点
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 矩形框的四个顶点相对于中心点的偏移
        half_width = width / 2
        half_length = length / 2
        
        # 四个顶点（相对于中心点）
        pts = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # 旋转并平移顶点
        rotated_pts = []
        for px, py in pts:
            # 旋转
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # 平移
            rotated_pts.append((int(center_x + rx), int(center_y + ry)))
        
        return rotated_pts
    
    def segment_intersection(self, A, B, C, D):
        """计算两条线段 AB 和 CD 的交点，若无交点则返回 None"""
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])

        if (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)):
            # 使用线段参数法计算交点
            xdiff = (A[0] - B[0], C[0] - D[0])
            ydiff = (A[1] - B[1], C[1] - D[1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                return None

            d = (det(A, B), det(C, D))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return (int(x), int(y))

        return None

    def find_intersections(self, box_pts, path):
        """找出矩形框与路径的所有交点"""
        intersections = []

        # 矩形框边（4条边）
        box_edges = [
            (box_pts[i], box_pts[(i + 1) % 4]) for i in range(4)
        ]

        for i in range(len(path) - 1):
            p1 = (path[i][1], path[i][0])  # 转为 (x, y)
            p2 = (path[i + 1][1], path[i + 1][0])

            for (bx1, by1), (bx2, by2) in box_edges:
                inter_pt = self.segment_intersection((bx1, by1), (bx2, by2), p1, p2)
                if inter_pt:
                    intersections.append((inter_pt[1], inter_pt[0]))  # 转回 (y, x)

        return intersections

    def animate_robot(self):
        """动画显示机器人沿路径移动"""
        if not self.smooth_path:
            print("没有可用的路径进行动画演示!")
            return
        
        # 创建放大的动画窗口
        cv2.namedWindow("Robot Animation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Animation", 
                        int(self.display_img.shape[1] * CONFIG["animation_scale"]),
                        int(self.display_img.shape[0] * CONFIG["animation_scale"]))
        
        # 获取机器人尺寸
        robot_h, robot_w = self.robot_img.shape[:2]
        
        # 创建放大后的副本用于动画
        anim_img = cv2.resize(self.display_img.copy(), None, 
                            fx=CONFIG["animation_scale"], fy=CONFIG["animation_scale"],
                            interpolation=cv2.INTER_LINEAR)
        
        # 绘制放大后的完整路径
        path_points = np.array([(x * CONFIG["animation_scale"], 
                                y * CONFIG["animation_scale"]) for (y, x) in self.smooth_path])
        cv2.polylines(anim_img, [path_points.astype(int)], False, (255, 0, 255), 2)
        
        # 添加速度控制条
        cv2.createTrackbar("Speed", "Robot Animation", self.animation_speed, 100, lambda x: None)
        
        # 放大起点和终点标记
        start_point = (int(self.start_point[1] * CONFIG["animation_scale"]),
                    int(self.start_point[0] * CONFIG["animation_scale"]))
        end_point = (int(self.end_point[1] * CONFIG["animation_scale"]),
                    int(self.end_point[0] * CONFIG["animation_scale"]))
        cv2.circle(anim_img, start_point, int(10 * CONFIG["animation_scale"]), (0, 255, 0), -1)
        cv2.circle(anim_img, end_point, int(10 * CONFIG["animation_scale"]), (0, 0, 255), -1)
        
        # 设置更大的字体和线条
        font_scale = CONFIG["animation_scale"] * 0.7
        thickness = int(CONFIG["animation_scale"])
        
        # 添加视频保存功能初始化
        save_video = CONFIG.get("save_animation", True)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可选: 'XVID'
            video_path = CONFIG.get("animation_video_path", "robot_animation.mp4")
            fps = CONFIG.get("animation_fps", 30)
            height, width = anim_img.shape[:2]
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        total_steps = len(self.smooth_path)
        
        for i, (y, x) in enumerate(self.smooth_path):
            # 检查窗口是否仍然存在
            if cv2.getWindowProperty("Robot Animation", cv2.WND_PROP_VISIBLE) < 1:
                break
                
            # 获取当前速度
            self.animation_speed = cv2.getTrackbarPos("Speed", "Robot Animation")
            delay = max(10, 100 - self.animation_speed)  # 速度映射到10-100ms
            
            # 清除上一帧
            frame = anim_img.copy()
            
            # 计算机器人角度和斜率
            angle = 0
            slope = 0
            if i < len(self.smooth_path)-1:
                next_y, next_x = self.smooth_path[i+1]
                angle = np.degrees(np.arctan2(next_x - x, next_y - y)) - 90
                slope = self.calculate_slope((y, x), (next_y, next_x))
            
            # 旋转机器人图像
            rotated_robot = self.rotate_image(self.robot_img, angle)
            rh, rw = rotated_robot.shape[:2]
            
            # 计算机器人位置(居中)
            robot_x = int(x * CONFIG["animation_scale"] - rw//2)
            robot_y = int(y * CONFIG["animation_scale"] - rh//2)
            
            # 将机器人叠加到地图上(带透明度)
            if rotated_robot.shape[2] == 4:  # 如果有alpha通道
                alpha = rotated_robot[:, :, 3] / 255.0
                for c in range(3):
                    # 确保坐标在有效范围内
                    y_start = max(0, robot_y)
                    y_end = min(frame.shape[0], robot_y + rh)
                    x_start = max(0, robot_x)
                    x_end = min(frame.shape[1], robot_x + rw)
                    
                    # 调整机器人图像的对应部分
                    r_y_start = max(0, -robot_y)
                    r_y_end = rh - max(0, (robot_y + rh) - frame.shape[0])
                    r_x_start = max(0, -robot_x)
                    r_x_end = rw - max(0, (robot_x + rw) - frame.shape[1])
                    
                    if y_end > y_start and x_end > x_start:
                        frame[y_start:y_end, x_start:x_end, c] = \
                            frame[y_start:y_end, x_start:x_end, c] * (1 - alpha[r_y_start:r_y_end, r_x_start:r_x_end]) + \
                            rotated_robot[r_y_start:r_y_end, r_x_start:r_x_end, c] * alpha[r_y_start:r_y_end, r_x_start:r_x_end]
            else:
                # 确保坐标在有效范围内
                y_start = max(0, robot_y)
                y_end = min(frame.shape[0], robot_y + rh)
                x_start = max(0, robot_x)
                x_end = min(frame.shape[1], robot_x + rw)
                
                if y_end > y_start and x_end > x_start:
                    frame[y_start:y_end, x_start:x_end] = rotated_robot[ 
                        max(0, -robot_y):rh - max(0, (robot_y + rh) - frame.shape[0]),
                        max(0, -robot_x):rw - max(0, (robot_x + rw) - frame.shape[1])
                    ]
            
            # 计算机器人前方的矩形框
            self.front_box_pts = self.calculate_front_box((y, x), -angle)
            scaled_box_pts = [(int(x * CONFIG["animation_scale"]), 
                            int(y * CONFIG["animation_scale"])) for (x, y) in self.front_box_pts]
            
            # 绘制矩形框
            cv2.polylines(frame, [np.array(scaled_box_pts, np.int32)], True, CONFIG["box_color"], 2)
            
            # 找出矩形框与路径的交点
            self.intersection_points = self.find_intersections(self.front_box_pts, self.smooth_path)
            
            # 绘制交点
            for pt in self.intersection_points:
                scaled_pt = (int(pt[1] * CONFIG["animation_scale"]), 
                            int(pt[0] * CONFIG["animation_scale"]))
                cv2.circle(frame, scaled_pt, CONFIG["intersection_size"], CONFIG["intersection_color"], -1)
            
            # 显示当前位置信息(放大后的字体)
            cv2.putText(frame, f"Position: ({x}, {y})", 
                        (int(20 * CONFIG["animation_scale"]), int(30 * CONFIG["animation_scale"])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, f"Step: {i}/{total_steps-1}", 
                        (int(20 * CONFIG["animation_scale"]), int(60 * CONFIG["animation_scale"])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, f"Slope: {slope:.2f}", 
                        (int(20 * CONFIG["animation_scale"]), int(90 * CONFIG["animation_scale"])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, f"Angle: {angle:.1f}°", 
                        (int(20 * CONFIG["animation_scale"]), int(120 * CONFIG["animation_scale"])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            cv2.putText(frame, f"Intersections: {len(self.intersection_points)}", 
                        (int(20 * CONFIG["animation_scale"]), int(150 * CONFIG["animation_scale"])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # 显示下一个目标点（如果有交点）
            if self.intersection_points:
                next_target = self.intersection_points[0]  # 取第一个交点
                cv2.putText(frame, f"Next Target: ({next_target[1]}, {next_target[0]})", 
                            (int(20 * CONFIG["animation_scale"]), int(180 * CONFIG["animation_scale"])),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
            
            cv2.imshow("Robot Animation", frame)
            
            # 保存视频
            if save_video:
                out.write(cv2.convertScaleAbs(frame))  # 保存当前帧

            # 更新进度条
            progress = int((i / total_steps) * 100)
            progress_bar = f"[{'█' * (progress // 5)}{' ' * (20 - (progress // 5))}] {progress}%"
            print(f"\r{progress_bar}", end="")
            
            key = cv2.waitKey(delay)
            
            if key == 27:  # ESC键退出动画
                break
        
        # 完成后释放视频文件并输出进度提示
        if save_video:
            out.release()
            print(f"\r{'[█████████████████████████] 100%'} 视频保存完成  -> {video_path}")
        
        cv2.waitKey(1000)  # 动画完成后暂停1秒
        cv2.destroyWindow("Robot Animation")

    @staticmethod
    def rotate_image(image, angle):
        """旋转图像并保持透明度"""
        if image.shape[2] == 4:  # 带alpha通道
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            return rotated
        else:  # 不带alpha通道
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
            return rotated

    def run_path_planning(self):
        """执行路径规划流程"""
        if not (self.start_point and self.end_point):
            print("错误：未设置起点或终点！")
            return
        
        start_time = time.time()
        raw_path = self.jps(self.inflated_map, self.start_point, self.end_point)
        
        if not raw_path:
            print("未找到有效路径！")
            return
        
        self.path = raw_path  # 保存原始路径
        
        # 路径平滑处理
        smooth_time_start = time.time()
        self.smooth_path = self.smooth_path_spline(raw_path, s=CONFIG["smoothing_factor"])
        smooth_time = time.time() - smooth_time_start
        
        total_elapsed_time = time.time() - start_time
        
        print(f"\n路径规划完成！")
        print(f"原始路径长度: {len(raw_path)} 个点")
        print(f"平滑路径长度: {len(self.smooth_path)} 个点")
        print(f"规划耗时: {total_elapsed_time:.4f} 秒 (平滑处理: {smooth_time:.4f} 秒)")
        
        # 保存结果
        self.save_results(raw_path, self.smooth_path)
        
        # 显示动画
        self.animate_robot()
        self.reset_selection()

    def save_results(self, raw_path, smooth_path):
        """保存可视化结果和路径数据"""
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        
        # 保存CSV数据
        csv_path = os.path.join(CONFIG["output_dir"], f"path_coords_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "raw_y", "raw_x", "smooth_y", "smooth_x"])
            
            # 确保两个路径可以一起迭代
            max_len = max(len(raw_path), len(smooth_path))
            for i in range(max_len):
                raw_y, raw_x = raw_path[min(i, len(raw_path)-1)] if raw_path else (-1, -1)
                smooth_y, smooth_x = smooth_path[min(i, len(smooth_path)-1)] if smooth_path else (-1, -1)
                writer.writerow([i, raw_y, raw_x, smooth_y, smooth_x])
                
        print(f"路径坐标已保存到: {csv_path}")
        
        # 可视化并保存图像
        self.visualize_results(raw_path, smooth_path, timestamp)

    def visualize_results(self, raw_path, smooth_path, timestamp):
        """可视化原始路径和平滑路径"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 原始地图可视化
        img_original = np.full((*self.binary_map.shape, 3), 255, dtype=np.uint8)
        img_original[self.binary_map == 0] = [50, 50, 50]  # 障碍物
        
        # 膨胀地图可视化
        img_inflated = img_original.copy()
        img_inflated[self.inflate_mask == 1] = [0, 0, 200]  # 膨胀区域
        
        # 提取坐标
        raw_y, raw_x = zip(*raw_path) if raw_path else ([], [])
        smooth_y, smooth_x = zip(*smooth_path) if smooth_path else ([], [])
        
        # 绘制路径
        ax1.imshow(img_inflated)
        ax1.plot(raw_x, raw_y, 'b-', linewidth=1, label='Raw Path')
        ax1.plot(smooth_x, smooth_y, 'r-', linewidth=2, label='Smooth Path')
        ax1.scatter(self.start_point[1], self.start_point[0], 
                  c='lime', s=200, edgecolors='black', label='Start')
        ax1.scatter(self.end_point[1], self.end_point[0],
                  c='magenta', s=200, edgecolors='black', label='End')
        ax1.legend()
        ax1.axis('off')
        
        ax2.imshow(img_original)
        ax2.plot(raw_x, raw_y, 'b-', linewidth=1, label='Raw Path')
        ax2.plot(smooth_x, smooth_y, 'r-', linewidth=2, label='Smooth Path')
        ax2.scatter(self.start_point[1], self.start_point[0], 
                  c='lime', s=200, edgecolors='black', label='Start')
        ax2.scatter(self.end_point[1], self.end_point[0],
                  c='magenta', s=200, edgecolors='black', label='End')
        ax2.legend()
        ax2.axis('off')
        
        ax1.set_title("Inflated Map with Path")
        ax2.set_title("Original Map with Path")
        
        # 保存图像
        img_path = os.path.join(CONFIG["output_dir"], f"path_result_{timestamp}.png")
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {img_path}")
        plt.show()
        plt.close()

    def reset_selection(self):
        """重置选择状态"""
        self.start_point = None
        self.end_point = None
        self.click_count = 0
        self.path = None
        self.smooth_path = None
        self.front_box_pts = None
        self.intersection_points = []

    def run(self):
        """主运行函数"""
        self.load_map()
        
        # 创建交互窗口
        cv2.namedWindow("Path Planning - Click to set start/end points")
        cv2.setMouseCallback("Path Planning - Click to set start/end points", 
                            self.on_mouse_click)
        
        # 显示操作说明
        self.display_img = cv2.cvtColor(self.binary_map, cv2.COLOR_GRAY2BGR)
        
        # cv2.putText(self.display_img, "1. Click to set START point", (20, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(self.display_img, "2. Click to set END point", (20, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(self.display_img, "ESC to exit", (20, 110),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        while self.running:
            temp_img = self.display_img.copy()
            if self.start_point:
                cv2.circle(temp_img, (self.start_point[1], self.start_point[0]), 
                          5, (0, 255, 0), -1)
            if self.end_point:
                cv2.circle(temp_img, (self.end_point[1], self.end_point[0]), 
                          5, (0, 0, 255), -1)
            
            cv2.imshow("Path Planning - Click to set start/end points", temp_img)
            
            # 检查窗口是否仍然存在
            if cv2.getWindowProperty("Path Planning - Click to set start/end points", cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
                
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC键退出
                self.running = False
                break
        
        cv2.destroyAllWindows()

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

if __name__ == "__main__":
    planner = PathPlanner()
    try:
        planner.run()
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()