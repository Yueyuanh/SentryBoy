import cv2
import numpy as np

# 全局变量存储坐标
start_point = None
end_point = None
click_count = 0

def on_mouse_click(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global start_point, end_point, click_count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
            start_point = (y, x)  # 存储为(y,x)格式
            print(f"起点设置: (y={y}, x={x})")
            click_count += 1
        elif click_count == 1:
            end_point = (y, x)
            print(f"终点设置: (y={y}, x={x})")
            click_count += 1

def select_points(map_path):
    """交互式选择起点和终点"""
    global start_point, end_point, click_count
    
    # 读取地图
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        print(f"错误：无法加载地图 {map_path}")
        return None, None
    
    # 创建彩色副本用于显示
    color_map = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    
    # 设置窗口和回调
    cv2.namedWindow("Select Points (Press ESC to quit)")
    cv2.setMouseCallback("Select Points (Press ESC to quit)", on_mouse_click)
    
    while True:
        # 实时显示当前选择状态
        display_map = color_map.copy()
        
        # 绘制已选择的点
        if start_point:
            cv2.circle(display_map, (start_point[1], start_point[0]), 5, (0, 255, 0), -1)
            cv2.putText(display_map, f"Start ({start_point[1]},{start_point[0]})", 
                       (start_point[1]+10, start_point[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if end_point:
            cv2.circle(display_map, (end_point[1], end_point[0]), 5, (0, 0, 255), -1)
            cv2.putText(display_map, f"End ({end_point[1]},{end_point[0]})", 
                       (end_point[1]+10, end_point[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 显示操作提示
        if click_count == 0:
            cv2.putText(display_map, "Click to set START point", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif click_count == 1:
            cv2.putText(display_map, "Click to set END point", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display_map, "Press ESC to confirm", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Select Points (Press ESC to quit)", display_map)
        
        # 按ESC键退出
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or click_count >= 2:  # ESC键或已选两个点
            break
    
    cv2.destroyAllWindows()
    return start_point, end_point

if __name__ == "__main__":
    # 使用示例
    MAP_PATH = "RMUC1.png"  # 替换为你的地图路径
    
    print("请在地图上依次点击选择起点和终点：")
    start, end = select_points(MAP_PATH)
    
    if start and end:
        print("\n最终选择：")
        print(f"起点坐标 (y,x): {start}")
        print(f"终点坐标 (y,x): {end}")
    else:
        print("未完成坐标选择！")