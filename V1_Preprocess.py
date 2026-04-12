import sys
import cv2
import numpy as np
import json
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QFormLayout,
                             QSizePolicy, QPushButton, QFileDialog, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from V0_ROI import auto_load_roi_masks, apply_roi_mask

# 定义文件名常量
CONFIG_FILENAME = "marker_params.json"

class MultiMarkerProSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("参数调整")
        self.setStyleSheet("QMainWindow { background-color: #ffffff; color: #000000; }")
        
        self.raw_imgs = []  # 存储原始图片列表
        self.processed_imgs = [] # 存储处理后的图片列表

        # [修改] 增加变量存储当前图片所在的文件夹路径
        self.current_img_folder = None
        # [修改] 增加字典存储滑块对象，以便加载配置时更新UI
        self.slider_widgets = {}

        # [新增] ROI掩膜存储
        self.left_roi_mask = None
        self.right_roi_mask = None

        # 默认参数
        self.params = {
            "blur": 3, "block_size": 11, "c_val": 5, "morph_size": 2,
            "min_area": 30, "max_area": 1000, "circularity": 70, "inertia": 30
        }
        
        # 初始化时不加载配置，改为在选择图片或手动加载时处理
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧控制面板 ---
        sidebar = QVBoxLayout()
        
        file_group = QGroupBox("文件与配置")
        file_group.setStyleSheet("QGroupBox { border: 1px solid #999; padding: 10px; color: #000; }")
        file_vbox = QVBoxLayout()
        
        btn_load = QPushButton("选择多张样张 (最多12张)")
        btn_load.clicked.connect(self.select_images)
        file_vbox.addWidget(btn_load)
        
        # [修改] 按钮文案提示更明确
        btn_save_cfg = QPushButton(f"保存参数到样张目录")
        btn_save_cfg.clicked.connect(self.save_config)
        file_vbox.addWidget(btn_save_cfg)

        btn_batch = QPushButton("开启批量处理")
        btn_batch.setStyleSheet("background-color: #0078d4; color: #ffffff; font-weight: bold;")
        btn_batch.clicked.connect(self.batch_processing)
        file_vbox.addWidget(btn_batch)
        
        file_group.setLayout(file_vbox)
        sidebar.addWidget(file_group)

        # 参数调节区
        ctrl_group = QGroupBox("算法参数同步调节")
        ctrl_group.setStyleSheet("QGroupBox { border: 1px solid #999; margin-top: 10px; padding: 10px; color: #000; }")
        self.form = QFormLayout()

        # 添加滑块
        self.add_slider("blur", "平滑 (Blur):", 1, 15)
        self.add_slider("block_size", "自适应窗口:", 3, 51)
        self.add_slider("c_val", "阈值补偿:", -10, 20)
        self.add_slider("morph_size", "形态学平滑:", 0, 10)
        self.add_slider("min_area", "最小面积:", 5, 500)
        self.add_slider("max_area", "最大面积:", 100, 1000)
        self.add_slider("circularity", "圆度过滤 %:", 0, 100)
        self.add_slider("inertia", "形状规整度 %:", 0, 100)

        ctrl_group.setLayout(self.form)
        sidebar.addWidget(ctrl_group)
        sidebar.addStretch()
        layout.addLayout(sidebar, 1)

        # --- 右侧 3x4 网格显示面板 ---
        display_container = QVBoxLayout()
        grid_layout = QGridLayout()

        self.display_labels = []
        for i in range(12):
            label = QLabel(f"样张 {i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; color: #666;")
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            grid_layout.addWidget(label, i // 4, i % 4)
            self.display_labels.append(label)
            
        display_container.addLayout(grid_layout, 1)
        
        self.info_label = QLabel("请加载样张图片")
        display_container.addWidget(self.info_label)
        layout.addLayout(display_container, 4)

    def add_slider(self, name, label, min_v, max_v):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        val = self.params.get(name, min_v)
        slider.setValue(val)
        # 避免初始化时大量触发计算，只有用户操作时触发
        slider.valueChanged.connect(lambda v: self.on_param_change(name, v))
        self.form.addRow(QLabel(label), slider)
        
        # [修改] 存储滑块引用，方便后续根据配置文件反向更新UI
        self.slider_widgets[name] = slider 
        return slider

    # [修改] 加载配置逻辑大改：根据当前目录加载
    def load_config(self):
        if not self.current_img_folder:
            return

        config_path = os.path.join(self.current_img_folder, CONFIG_FILENAME)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_params = json.load(f)
                    self.params.update(loaded_params)
                    
                    # [关键修改] 加载文件后，必须同步更新界面上的滑块位置
                    # block signals 防止更新滑块时重复触发图像处理
                    for key, val in loaded_params.items():
                        if key in self.slider_widgets:
                            slider = self.slider_widgets[key]
                            slider.blockSignals(True) 
                            slider.setValue(val)
                            slider.blockSignals(False)
                    
                    print(f"已从 {config_path} 加载参数")
                    self.info_label.setText(f"已加载目录配置文件: {CONFIG_FILENAME}")
            except Exception as e:
                print(f"配置加载失败: {e}")
        else:
            self.info_label.setText("当前目录无配置文件，使用默认参数")

    # [修改] 保存配置逻辑大改：保存到当前目录
    def save_config(self):
        if not self.current_img_folder:
            QMessageBox.warning(self, "警告", "请先选择样张图片，以便确定配置文件保存路径。")
            return

        save_path = os.path.join(self.current_img_folder, CONFIG_FILENAME)
        try:
            with open(save_path, 'w') as f:
                json.dump(self.params, f, indent=4) # indent让json更易读
            QMessageBox.information(self, "成功", f"参数已保存至:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    # [新增] 加载ROI掩膜
    def load_roi_masks(self):
        """从当前图片目录自动加载ROI掩膜"""
        if not self.current_img_folder:
            return

        left_mask, right_mask = auto_load_roi_masks(self.current_img_folder)
        if left_mask is not None and right_mask is not None:
            self.left_roi_mask = left_mask
            self.right_roi_mask = right_mask
            self.info_label.setText(f"已加载ROI掩膜")
            print(f"✓ 已加载ROI掩膜")
        else:
            self.left_roi_mask = None
            self.right_roi_mask = None
            self.info_label.setText("未找到ROI掩膜，将检测全图")
            print("⚠ 未找到ROI掩膜文件")

    def on_param_change(self, name, value):
        # 参数约束逻辑
        if name == "block_size" and value % 2 == 0: value += 1
        if name == "blur" and value % 2 == 0: value += 1
        
        self.params[name] = value
        self.process_all_images()

    def select_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, '选取样张 (按住Ctrl多选)', '', 'Image files (*.jpg *.png *.bmp)')
        if fnames:
            # [修改] 获取第一张图片的目录作为当前工作目录
            first_img_path = Path(fnames[0])
            self.current_img_folder = str(first_img_path.parent)

            # [修改] 切换目录后，尝试自动加载该目录下的配置文件
            self.load_config()

            # [新增] 自动加载ROI掩膜
            self.load_roi_masks()

            self.raw_imgs = []
            for path in fnames[:12]: # 最多取前12张
                img = cv2.imread(path)
                if img is not None:
                    self.raw_imgs.append(img)

            # 加载新图片后立即处理
            self.process_all_images()

    def _detect_by_local_minima(self, gray, blur):
        """基于局部极小值的检测：补救密集区域丢失的点"""
        from scipy.spatial import cKDTree
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

    def detection_logic(self, img):
        """核心检测算法（使用局部极小值检测）"""
        if img is None: return None, 0

        # 分离左右图像
        h, w = img.shape[:2]
        mirror_axis = w // 2

        # 创建左右图像
        left_img = np.full((h, w, 3), 255, dtype=np.uint8)
        left_img[:, :mirror_axis] = img[:, :mirror_axis]
        right_img = np.full((h, w, 3), 255, dtype=np.uint8)
        right_img[:, mirror_axis:] = img[:, mirror_axis:]

        # 分别检测左右图像
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # CLAHE 局部对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        left_gray = clahe.apply(left_gray)
        right_gray = clahe.apply(right_gray)

        # 平滑滤波 — 限制最大blur防止密集点被合并
        b_val = max(1, self.params["blur"])
        if b_val % 2 == 0: b_val += 1
        b_val = min(b_val, 7)
        left_blur = cv2.GaussianBlur(left_gray, (b_val, b_val), 0)
        right_blur = cv2.GaussianBlur(right_gray, (b_val, b_val), 0)

        # 使用局部极小值检测 - 分别处理左右图像
        res_img = img.copy()
        left_minima_points = self._detect_by_local_minima(left_gray, left_blur)
        right_minima_points = self._detect_by_local_minima(right_gray, right_blur)

        # 应用ROI掩膜过滤
        if self.left_roi_mask is not None:
            left_minima_points = apply_roi_mask(left_minima_points, self.left_roi_mask)

        if self.right_roi_mask is not None:
            right_minima_points = apply_roi_mask(right_minima_points, self.right_roi_mask)

        # 合并左右结果并去重
        all_points = left_minima_points + right_minima_points
        min_area = self.params["min_area"]
        # 增大去重半径到3倍，确保圆圈不会交叉
        # 如果两个圆的圆心距离小于 (半径1 + 半径2)，它们就会交叉
        # 所以去重距离应该至少是 2 * 平均半径
        avg_radius = int(np.sqrt(min_area / np.pi))
        dedup_radius = max(8, avg_radius * 2.5)  # 使用3倍半径作为去重距离
        all_points = self._remove_duplicate_points(all_points, dedup_radius)

        # 绘制结果
        for (x, y, r) in all_points:
            cv2.circle(res_img, (x, y), r + 2, (0, 255, 0), 2)

        return res_img, len(all_points)

    def _remove_duplicate_points(self, points, min_distance):
        """移除距离过近的重复点，只保留一个"""
        if len(points) <= 1:
            return points

        from scipy.spatial import cKDTree
        points_arr = np.array([(x, y) for x, y, _ in points], dtype=np.float32)
        tree = cKDTree(points_arr)

        # 找出所有距离小于min_distance的点对
        pairs = tree.query_pairs(min_distance)

        # 标记要删除的点
        to_remove = set()
        for i, j in pairs:
            # 保留索引较小的点（先检测到的），删除索引较大的点
            to_remove.add(j)

        # 返回未被标记删除的点
        result = [points[i] for i in range(len(points)) if i not in to_remove]
        return result

    def process_all_images(self):
        if not self.raw_imgs: return
        self.processed_imgs = []
        counts = []
        for img in self.raw_imgs:
            p_img, c = self.detection_logic(img)
            self.processed_imgs.append(p_img)
            counts.append(c)
        
        # 显示路径信息和计数
        path_info = " (已加载配置)" if os.path.exists(os.path.join(str(self.current_img_folder), CONFIG_FILENAME)) else ""
        self.info_label.setText(f"检测点数: {counts}{path_info}")
        self.update_all_displays()

    def update_all_displays(self):
        for i in range(12):
            if i < len(self.processed_imgs):
                img = self.processed_imgs[i]
                if img is None: continue

                target_size = self.display_labels[i].size()
                if target_size.width() > 5:
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    pixmap = QPixmap.fromImage(q_img).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.display_labels[i].setPixmap(pixmap)
            else:
                self.display_labels[i].clear()
                self.display_labels[i].setText(f"未加载样张 {i+1}")

    def batch_processing(self):
        src_dir = QFileDialog.getExistingDirectory(self, "选择批量处理文件夹")
        if not src_dir: return
        src_path = Path(src_dir)
        save_dir = src_path.parent / f"{src_path.name}_Processed"
        if not save_dir.exists(): save_dir.mkdir()
        
        # [修改] 批量处理时，也尝试读取该文件夹下的配置，以保证批量效果的一致性
        # 如果存在配置文件，临时加载该参数用于处理
        batch_config = src_path / CONFIG_FILENAME
        temp_params = self.params.copy() # 备份当前UI参数
        if batch_config.exists():
             with open(batch_config, 'r') as f:
                self.params.update(json.load(f))
        
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        if not files: return
        
        count_processed = 0
        for f_name in files:
            img = cv2.imread(os.path.join(src_dir, f_name))
            if img is not None:
                res_img, _ = self.detection_logic(img)
                cv2.imwrite(os.path.join(str(save_dir), f_name), res_img)
                count_processed += 1
        
        # 恢复UI参数（如果需要的话，或者保留刚刚加载的参数）
        # self.params = temp_params 
        
        QMessageBox.information(self, "完成", f"处理了 {count_processed} 张图片。\n保存在: {save_dir}")

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.update_all_displays()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiMarkerProSuite()
    window.resize(1280, 850)
    window.show()
    sys.exit(app.exec_())