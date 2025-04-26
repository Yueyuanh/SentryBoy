import cv2
import argparse

def resize_image(image_path, output_path, width=None, height=None, inter=cv2.INTER_AREA):
    """
    等比缩放图像
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    :param width: 目标宽度（可选）
    :param height: 目标高度（可选）
    :param inter: 插值方法（默认cv2.INTER_AREA）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像 {image_path}")
    
    # 获取原始尺寸
    (h, w) = image.shape[:2]
    
    # 如果未指定任何尺寸，返回原图
    if width is None and height is None:
        print("未指定缩放尺寸，将保存原图")
        cv2.imwrite(output_path, image)
        return
    
    # 根据指定的宽度或高度计算缩放比例
    if width is None:
        # 只指定高度的情况
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        # 只指定宽度的情况
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    
    # 执行缩放
    resized = cv2.resize(image, dim, interpolation=inter)
    
    # 保存结果
    cv2.imwrite(output_path, resized)
    print(f"图像已缩放: {w}x{h} -> {dim[0]}x{dim[1]} (保存到 {output_path})")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="图像等比缩放工具")
    parser.add_argument("-i", "--input", required=True, help="输入图像路径")
    parser.add_argument("-o", "--output", required=True, help="输出图像路径")
    parser.add_argument("-w", "--width", type=int, help="目标宽度（像素）")
    parser.add_argument("-ht", "--height", type=int, help="目标高度（像素）")
    args = parser.parse_args()

    # 执行缩放
    try:
        resize_image(args.input, args.output, args.width, args.height)
    except Exception as e:
        print(f"错误: {str(e)}")


# # 指定宽度（高度自动计算）
# python resize.py -i input.jpg -o output.jpg -w 800

# # 指定高度（宽度自动计算）
# python resize.py -i input.jpg -o output.jpg -ht 600

# # 同时指定宽高（不保持比例）
# python resize.py -i input.jpg -o output.jpg -w 800 -ht 600