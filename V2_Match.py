# -*- coding: utf-8 -*-
'''
标定辅助工具：点对匹配与数据保存（批量匹配增强版）
功能：
1. 批量匹配模式：左图点选队列 -> 右图顺序对齐
2. 撤销功能：按 'z' 键撤销队列点或已完成的匹配对
3. ROI功能已迁移至V0_ROI.py模块
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from V0_ROI import draw_polygon_roi, apply_roi_mask, save_roi_masks, load_roi_masks, get_roi_file_path

# --- 1. 标定参数与配置路径 ---
CALIB_DIR = "./calibration_1mm_12X9_0416_Paras"
CONFIG_FILE = "marker_params.json"  # 这里只保留文件名

# 全局变量存储绘制状态
drawing = {
    'left': {'points': [], 'mask': None},
    'right': {'points': [], 'mask': None}
}

# -------------------- 核心检测类 --------------------
class CircleDetector:
    def __init__(self, config_path="marker_params.json", verbose=True):
        # 默认参数
        self.params = {
            "blur": 3, "block_size": 11, "c_val": 5, "morph_size": 2,
            "min_area": 30, "max_area": 1000, "circularity": 70, "inertia": 30
        }
        # 加载参数
        import os, json
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_params = json.load(f)
                    self.params.update(saved_params)
                if verbose:
                    print(f"✓ 已加载参数配置: {config_path}")
            except Exception as e:
                if verbose:
                    print(f"⚠ 加载参数文件失败，使用默认值: {e}")
        else:
            if verbose:
                print("⚠ 未找到 marker_params.json，使用系统默认参数")

    def _detect_by_local_minima(self, gray, blur):
        import cv2
        import numpy as np
        
        min_area = self.params["min_area"]
        avg_radius = max(3, int(np.sqrt(min_area / np.pi)))
        win = avg_radius * 2 + 1

        smooth = cv2.GaussianBlur(blur, (3, 3), 0)
        local_min = cv2.erode(smooth, np.ones((win, win), np.uint8))
        minima_mask = (smooth == local_min).astype(np.uint8) * 255

        block = max(7, win * 3 | 1)
        dark_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, block, self.params["c_val"])
        minima_mask = cv2.bitwise_and(minima_mask, dark_mask)

        dilate_k = max(2, avg_radius)
        minima_mask = cv2.dilate(minima_mask, np.ones((dilate_k, dilate_k), np.uint8))

        contours, _ = cv2.findContours(minima_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                points.append((cx, cy, avg_radius))
        return points

    def _remove_duplicate_points(self, points, min_distance):
        """移除距离过近的重复点，只保留一个"""
        if len(points) <= 1:
            return points

        import numpy as np
        from scipy.spatial import cKDTree
        
        points_arr = np.array([(x, y) for x, y, _ in points], dtype=np.float32)
        tree = cKDTree(points_arr)

        # 找出所有距离小于min_distance的点对
        pairs = tree.query_pairs(min_distance)

        # 标记要删除的点
        to_remove = set()
        for i, j in pairs:
            to_remove.add(j)

        # 返回未被标记删除的点
        result = [points[i] for i in range(len(points)) if i not in to_remove]
        return result

    def detect(self, image):
        import cv2
        import numpy as np
        
        # 1. 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHE 局部对比度增强，抵抗受压后亮度变化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 平滑滤波 — 限制最大blur防止密集点被合并
        b_val = max(1, self.params.get("blur", 3))
        if b_val % 2 == 0: b_val += 1
        b_val = min(b_val, 7)
        blur = cv2.GaussianBlur(gray, (b_val, b_val), 0)

        # 2. 局部极小值检测 (完全替代原本的轮廓检测)
        points = self._detect_by_local_minima(gray, blur)

        # 3. KDTree 去重
        min_area = self.params["min_area"]
        avg_radius = int(np.sqrt(min_area / np.pi))
        dedup_radius = max(8, avg_radius * 2.5)  # 2.5倍半径作为去重距离
        points = self._remove_duplicate_points(points, dedup_radius)

        return points

# -------------------- 核心业务流程 --------------------

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择图像文件",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    return file_path

def setup_io_paths(image_path):
    base_dir = os.path.dirname(os.path.abspath(image_path))
    data_dir = os.path.join(base_dir, "data")
    result_dir = os.path.join(base_dir, "result")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    paths = {
        'roi_mask': os.path.join(data_dir, 'roi_masks.npz'),
        'match_data': os.path.join(data_dir, 'matched_points.npz'),
        'output_points': os.path.join(data_dir, 'output_points'),
        'result_dir': result_dir  # V3风格的结果目录
    }
    return paths

def process_and_save(image_path, calib_data, io_paths):
    K1, D1, K2, D2 = calib_data['K1'], calib_data['D1'], calib_data['K2'], calib_data['D2']
    R, T = calib_data['R'], calib_data['T']

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法加载图片: {image_path}")
    
    h, w = img.shape[:2]
    mirror_axis = w // 2

    left_img = np.full((h, w, 3), 255, dtype=np.uint8)
    left_img[:, :mirror_axis] = img[:, :mirror_axis]
    right_img = np.full((h, w, 3), 255, dtype=np.uint8)
    right_img[:, mirror_axis:] = img[:, mirror_axis:]

    roi_file = io_paths['roi_mask']
    left_mask, right_mask = load_roi_masks(roi_file)
    if left_mask is not None and right_mask is not None:
        drawing['left']['mask'] = left_mask
        drawing['right']['mask'] = right_mask
    else:
        drawing['left']['mask'] = draw_polygon_roi(left_img.copy(), 'left')
        drawing['right']['mask'] = draw_polygon_roi(right_img.copy(), 'right')
        if drawing['left']['mask'] is not None and drawing['right']['mask'] is not None:
            save_roi_masks(drawing['left']['mask'], drawing['right']['mask'], roi_file)

    # [修改核心] 获取图像所在的目录，并拼接 json 路径
    image_dir = os.path.dirname(os.path.abspath(image_path))
    target_config_path = os.path.join(image_dir, CONFIG_FILE)
    
    # 实例化检测器时传入该路径
    detector = CircleDetector(config_path=target_config_path)
    
    raw_left = detector.detect(left_img)
    raw_right = detector.detect(right_img)
    
    left_circles = apply_roi_mask(raw_left, drawing['left']['mask'])
    right_circles = apply_roi_mask(raw_right, drawing['right']['mask'])
    print(f"检测到点数 - 左: {len(left_circles)}, 右: {len(right_circles)}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
    
    left_pts_arr = np.array([(x, y) for (x, y, _) in left_circles], dtype=np.float32)
    right_pts_arr = np.array([(x, y) for (x, y, _) in right_circles], dtype=np.float32)

    left_undistorted = cv2.undistortPoints(left_pts_arr.reshape(-1,1,2), K1, D1, R=R1, P=P1).reshape(-1,2) if len(left_pts_arr) > 0 else np.array([])
    right_undistorted = cv2.undistortPoints(right_pts_arr.reshape(-1,1,2), K2, D2, R=R2, P=P2).reshape(-1,2) if len(right_pts_arr) > 0 else np.array([])

    matched_pairs = []
    match_file = io_paths['match_data']
    if os.path.exists(match_file):
        try:
            with np.load(match_file) as data:
                matched_pairs = data['matched_pairs'].tolist()
        except: pass

    if not matched_pairs and len(left_circles) > 0 and len(right_circles) > 0:
        matched_pairs = manual_point_matching(left_img, right_img, [p[:2] for p in left_circles], [p[:2] for p in right_circles])
        if len(matched_pairs) > 0:
            np.savez(match_file, left_points=np.array(left_circles), right_points=np.array(right_circles), 
                     matched_pairs=np.array(matched_pairs, dtype=np.int32), mirror_axis=mirror_axis, image_shape=np.array([h, w]))

    if len(matched_pairs) > 0:
        pts1 = left_undistorted[[i for i, j in matched_pairs]]
        pts2 = right_undistorted[[j for i, j in matched_pairs]]
        points_3d = linear_triangulation(pts1, pts2, P1, P2)

        # 获取匹配后的原始2D点坐标（用于V3格式保存）
        left_pts_matched = left_pts_arr[[i for i, j in matched_pairs]]
        right_pts_matched = right_pts_arr[[j for i, j in matched_pairs]]

        # 按V3格式保存到result目录
        save_first_frame_result(points_3d, left_pts_matched, right_pts_matched, io_paths['result_dir'])

        # 原有的保存方式（保留兼容）
        save_points_to_txt(points_3d, io_paths['output_points'])
        plot_3d_points(points_3d)
    else:
        print("提示: 未进行匹配或无有效匹配点。")

# -------------------- 辅助函数 --------------------

def manual_point_matching(left_img, right_img, left_points, right_points):
    matches = []
    left_queue = []  # 左图预选队列
    h, full_w = left_img.shape[:2]
    split_point = full_w // 2
    effective_left = left_img[:, :split_point]
    effective_right = right_img[:, split_point:]
    composite = np.hstack((effective_left, effective_right))
    left_pts_dict = {i: (int(p[0]), int(p[1])) for i, p in enumerate(left_points)}
    right_pts_dict = {j: (int(p[0]), int(p[1])) for j, p in enumerate(right_points)}
    radius = 6

    def draw_interface():
        display = composite.copy()
        # 绘制左侧点
        for i, (x, y) in left_pts_dict.items():
            if any(i == m[0] for m in matches):
                color = (0, 255, 0) # 已匹配：绿色
            elif i in left_queue:
                color = (0, 255, 255) # 在队列中：黄色
            else:
                color = (255, 0, 0) # 未处理：蓝色
            cv2.circle(display, (x, y), radius, color, 2)
            if i in left_queue:
                order = left_queue.index(i) + 1
                cv2.putText(display, str(order), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制右侧点
        for j, (x, y) in right_pts_dict.items():
            disp_x = x # 已经是全局坐标
            color = (0, 255, 0) if any(j == m[1] for m in matches) else (255, 0, 0)
            cv2.circle(display, (disp_x, y), radius, color, 2)
        return display

    def check_and_select_point(x, y):
        """检查坐标附近是否有点，有则选中"""
        nonlocal matches, left_queue
        if x < split_point:  # 左半区：加入待匹配队列
            for pid, (px, py) in left_pts_dict.items():
                if np.sqrt((x-px)**2 + (y-py)**2) < radius*1.5:
                    if pid not in left_queue and not any(m[0] == pid for m in matches):
                        left_queue.append(pid)
                    break
        else:  # 右半区：自动按顺序与队列首位匹配
            for pid, (px, py) in right_pts_dict.items():
                if np.sqrt((x-px)**2 + (y-py)**2) < radius*1.5:
                    if any(m[1] == pid for m in matches): break
                    if len(left_queue) > 0:
                        left_pid = left_queue.pop(0)
                        matches.append((left_pid, pid))
                    break

    def mouse_cb(event, x, y, flags, param):
        # 点击时选中
        if event == cv2.EVENT_LBUTTONDOWN:
            check_and_select_point(x, y)
        # 按住左键拖动时也选中（滑过即选）
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            check_and_select_point(x, y)

    cv2.namedWindow("Manual Matching", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Matching", mouse_cb)
    print("\n【操作说明】")
    print("1. 在左图点击或按住鼠标滑过圆点即可选中（黄色数字表示顺序）")
    print("2. 在右图点击或按住鼠标滑过圆点，程序自动按顺序配对")
    print("3. 按 'z' 撤销，'s' 保存退出，'c' 清空，ESC 取消\n")

    while True:
        cv2.imshow("Manual Matching", draw_interface())
        key = cv2.waitKey(1)
        if key == ord('s'): break
        elif key == ord('z'): # 优先撤销队列，再撤销已匹配对
            if left_queue:
                left_queue.pop()
            elif matches:
                matches.pop()
        elif key == ord('c'): 
            matches = []; left_queue = []
        elif key == 27: 
            matches = []; break
    cv2.destroyAllWindows()
    return matches

# ROI相关函数已迁移至V0_ROI.py模块

def linear_triangulation(pts1, pts2, P1, P2):
    num_points = pts1.shape[0]
    points_3d = np.zeros((num_points, 3))
    for i in range(num_points):
        x1, y1 = pts1[i]; x2, y2 = pts2[i]
        A = np.array([x1*P1[2,:]-P1[0,:], y1*P1[2,:]-P1[1,:], x2*P2[2,:]-P2[0,:], y2*P2[2,:]-P2[1,:]])
        _, _, V = np.linalg.svd(A); X = V[-1, :]; points_3d[i] = X[:3] / X[3]
    return points_3d

def load_stereo_params(param_dir):
    p = {}
    try:
        for n in ['K1','K2','R']: p[n] = np.loadtxt(f"{param_dir}/{n}.txt").reshape(3,3)
        for n in ['D1','D2']: p[n] = np.loadtxt(f"{param_dir}/{n}.txt").reshape(1,-1)
        p['T'] = np.loadtxt(f"{param_dir}/T.txt").reshape(3,1)
    except Exception as e:
        print(f"加载标定参数失败: {e}"); return None
    return p

def save_points_to_txt(points_3d, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'reconstructed_points.txt')
    np.savetxt(save_path, points_3d, fmt='%.3f', delimiter='\t', header='X(mm)\tY(mm)\tZ(mm)')
    print(f"数据已保存至: {save_path}")

def save_first_frame_result(points_3d, left_points, right_points, result_dir):
    """
    按V3格式保存第一帧结果
    格式: X(mm) Y(mm) Z(mm) Left_x Left_y Right_x Right_y
    """
    import shutil

    # 如果result目录存在则先清空
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # 合并数据: 3D坐标 + 左侧2D + 右侧2D
    combined_data = np.hstack((points_3d, left_points, right_points))
    header = 'X(mm)\tY(mm)\tZ(mm)\tLeft_x\tLeft_y\tRight_x\tRight_y'

    # 保存为frame_000_points.txt
    filename = os.path.join(result_dir, 'frame_000_points.txt')
    np.savetxt(filename, combined_data, fmt='%.3f', delimiter='\t', header=header, comments='')

    print(f"V3格式结果已保存至: {filename}")

def plot_3d_points(points_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='r', marker='o')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    plt.title("3D Reconstructed Markers")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(CALIB_DIR):
        print(f"错误: 找不到标定参数目录 {CALIB_DIR}")
    else:
        selected_image = select_image_file()
        if selected_image:
            calib_params = load_stereo_params(CALIB_DIR)
            if calib_params:
                io_paths = setup_io_paths(selected_image)
                process_and_save(selected_image, calib_params, io_paths)