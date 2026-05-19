# -*- coding: utf-8 -*-
'''
ROI区域管理模块
功能：
1. 绘制多边形ROI区域
2. 保存和加载ROI掩膜
3. 应用ROI掩膜过滤检测点
'''

import cv2
import numpy as np
import os


def draw_polygon_roi(img, side):
    """
    交互式绘制多边形ROI区域

    参数:
        img: 输入图像
        side: 标识符（如'left'或'right'）

    返回:
        mask: ROI掩膜（numpy数组），如果点数不足则返回None

    操作说明:
        - 左键点击：添加ROI顶点
        - 中键点击：撤销最后一个顶点
        - 按'q'键：完成绘制
    """
    window_name = f"Draw ROI - {side}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    pts_list = []

    def mouse_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_list.append((x, y))
        elif event == cv2.EVENT_MBUTTONDOWN and len(pts_list) > 0:
            pts_list.pop()

    cv2.setMouseCallback(window_name, mouse_roi)
    print(f"正在绘制 {side} ROI: 左键点选，中键撤销，'q'完成。")

    while True:
        disp = img.copy()
        if len(pts_list) >= 2:
            cv2.polylines(disp, [np.array(pts_list, np.int32)], False, (0, 255, 255), 2)
        for pt in pts_list:
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
        cv2.imshow(window_name, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mask = None
    if len(pts_list) >= 3:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts_list, np.int32)], 255)

    cv2.destroyAllWindows()
    return mask


def save_roi_masks(left_mask, right_mask, save_path):
    """
    保存左右两侧的ROI掩膜到文件

    参数:
        left_mask: 左侧ROI掩膜
        right_mask: 右侧ROI掩膜
        save_path: 保存路径（.npz文件）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, left_mask=left_mask, right_mask=right_mask)
    print(f"ROI掩膜已保存至: {save_path}")


def load_roi_masks(load_path):
    """
    从文件加载ROI掩膜

    参数:
        load_path: ROI掩膜文件路径（.npz文件）

    返回:
        (left_mask, right_mask): 左右两侧的ROI掩膜，如果加载失败返回(None, None)
    """
    if not os.path.exists(load_path):
        print(f"ROI文件不存在: {load_path}")
        return None, None

    try:
        with np.load(load_path) as data:
            if 'left_mask' not in data or 'right_mask' not in data:
                print(f"ROI文件格式错误: 缺少必要的数据")
                return None, None

            left_mask = data['left_mask']
            right_mask = data['right_mask']

            # 处理可能的多通道掩膜
            if left_mask.ndim == 3:
                left_mask = left_mask[:, :, 0]
            if right_mask.ndim == 3:
                right_mask = right_mask[:, :, 0]

            print(f"成功加载ROI掩膜: {load_path}")
            return left_mask, right_mask
    except Exception as e:
        print(f"加载ROI掩膜失败: {e}")
        return None, None


def apply_roi_mask(points, mask):
    """
    应用ROI掩膜过滤检测点

    参数:
        points: 检测到的点列表，格式为[(x, y, r), ...]
        mask: ROI掩膜（numpy数组）

    返回:
        valid_points: 在ROI区域内的点列表
    """
    if mask is None:
        return points

    valid = []
    for (x, y, r) in points:
        if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1]:
            if mask[int(y), int(x)] > 0:
                valid.append((x, y, r))
    return valid


def get_roi_file_path(image_path_or_dir):
    """
    根据图像路径或目录获取ROI文件的标准路径

    参数:
        image_path_or_dir: 图像文件路径或图像所在目录

    返回:
        roi_file_path: ROI掩膜文件的完整路径
    """
    if os.path.isfile(image_path_or_dir):
        image_dir = os.path.dirname(os.path.abspath(image_path_or_dir))
    else:
        image_dir = os.path.abspath(image_path_or_dir)

    data_dir = os.path.join(image_dir, "data")
    roi_file = os.path.join(data_dir, "roi_masks.npz")
    return roi_file


def auto_load_roi_masks(image_path_or_dir):
    """
    自动从标准位置加载ROI掩膜

    参数:
        image_path_or_dir: 图像文件路径或图像所在目录

    返回:
        (left_mask, right_mask): 左右两侧的ROI掩膜，如果加载失败返回(None, None)
    """
    roi_file = get_roi_file_path(image_path_or_dir)
    return load_roi_masks(roi_file)


def create_roi_interactive(image_path):
    """
    交互式创建ROI掩膜并保存

    参数:
        image_path: 图像文件路径

    返回:
        success: 是否成功创建并保存ROI
    """
    # 读取图像
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误: 无法加载图像 {image_path}")
        return False

    print("\n=== ROI区域绘制工具 ===")
    print("即将依次绘制左侧和右侧的ROI区域")
    print("操作说明：")
    print("  - 左键点击：添加ROI顶点")
    print("  - 中键点击：撤销最后一个顶点")
    print("  - 按'q'键：完成当前ROI绘制\n")

    # 绘制左侧ROI
    print("步骤1: 绘制左侧ROI区域...")
    left_mask = draw_polygon_roi(img.copy(), 'left')
    if left_mask is None:
        print("错误: 左侧ROI绘制失败（顶点数不足3个）")
        return False

    # 绘制右侧ROI
    print("\n步骤2: 绘制右侧ROI区域...")
    right_mask = draw_polygon_roi(img.copy(), 'right')
    if right_mask is None:
        print("错误: 右侧ROI绘制失败（顶点数不足3个）")
        return False

    # 保存ROI掩膜
    roi_file = get_roi_file_path(image_path)
    save_roi_masks(left_mask, right_mask, roi_file)

    print(f"\n✓ ROI创建成功！")
    print(f"  保存位置: {roi_file}")
    return True


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    print("=== V0_ROI - ROI区域创建工具 ===\n")

    # 选择图像文件
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="请选择图像文件（镜像双目图像）",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not image_path:
        print("未选择文件，程序退出")
    else:
        print(f"已选择图像: {image_path}\n")
        success = create_roi_interactive(image_path)
        if success:
            print("\n程序执行完成！")
        else:
            print("\n程序执行失败！")
