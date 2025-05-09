# Sentry Boy
RMU 2D Sentry Nav Demo
![](Astar/results/path_result_20250426_010253.png)

# Animation
![](Astar/robot_animation.gif)

# Requirements
```
python-opencv
numpy
matplotlib
heapq
scipy
```
# Usage
```
python Astar_Plan.py
```

# Config
```
CONFIG = {
    "map_path": "map/RMUC1_B.png",      #地图
    "robot_path": "robot/sentry1.png",  # 机器人图片路径
    "inflate_radius": 10,               # 膨胀半径
    "output_dir": "results",            # 输出目录
    "default_speed": 1,                 # 默认移动速度(ms/点)
    "animation_scale": 2.0,             # 动画窗口放大倍数
    "smoothing_factor": 0.5,            # 路径平滑因子(0-1)，越大越平滑
    "spline_points": 200,               # 样条曲线插值点数
    "front_box_length": 30,             # 前方矩形框长度(像素)
    "front_box_width": 25,              # 前方矩形框宽度(像素)
    "box_color": (255, 165, 0),         # 矩形框颜色(橙色)
    "intersection_color": (255, 0, 255),# 交点颜色(紫色)
    "intersection_size": 10,            # 交点标记大小

    "save_animation": True,             # 是否录制动画
    "animation_video_path": "robot_animation.mp4",# 动画输出
    "animation_fps": 15,                # 动画帧率
}

```