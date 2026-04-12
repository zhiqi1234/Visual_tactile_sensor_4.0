# -*- coding: utf-8 -*-
'''
多帧标记点跟踪系统 (融合版)
集成说明：
1. 标记点检测算法已替换为：自适应阈值 + 形态学处理 + 形状约束过滤 (来自 Code 1)
2. 参数控制：自动读取 marker_params.json，移除了 GUI 中的冗余参数控件
3. 保持了原有的 GUI 框架、3D 重建、匹配和后处理逻辑
'''
import io
import sys
import os
import shutil
import threading
import time
import json  # 新增：用于读取配置文件

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QTabWidget, QTextEdit, QGroupBox, QStatusBar, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFormLayout, QSlider)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import Rbf, griddata
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class MarkerTrackingSystem(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "三维重建"
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1400, 900)

        # 全局状态
        self.calib_loaded = False
        self.first_frame_processed = False
        self.current_frame = 0

        # 记录当前工作的目录，用于自动定位 result 和 error 文件夹
        self.working_directory = None

        # 全局数据
        self.FRAME_DATA = {
            'initialized': False,
            'roi_masks': None,
            'P1': None,
            'P2': None,
            'left_points_0': None,
            'right_points_0': None,
            'left_points_0_R': None,
            'right_points_0_R': None,
            'left_points_0_pre': None,
            'right_points_0_pre': None,
            'base_3d_points': None,
            'mirror_axis': None,
            'transform_origin': None,
            'transform_rotation': None,
            'reference_surface': None
        }
        self.ALL_FRAME_POINTS = []
        self.drawing = {
            'left': {'points': [], 'mask': None, 'current': False},
            'right': {'points': [], 'mask': None, 'current': False}
        }
        # 全局变量保存帧的有效性 (1有效, 0无效)
        self.frame_validity = []

        # 缓存检测器实例，避免重复创建和加载配置文件
        self.detector = None

        # 保存3D视图角度的变量
        # 点云视图角度（首帧处理后的3D点云）
        self.pointcloud_view_elev = None
        self.pointcloud_view_azim = None
        self.pointcloud_view_roll = 0  # matplotlib 3.6+ 支持 roll
        self.pointcloud_view_saved = False  # 标记用户是否手动调整过角度

        # 后处理三维表面视图角度
        self.surface_view_elev = None
        self.surface_view_azim = None
        self.surface_view_roll = 0  # matplotlib 3.6+ 支持 roll
        self.surface_view_saved = False  # 标记用户是否手动调整过角度

        # 标记用户是否正在拖动3D视图（防止拖动时被刷新覆盖）
        self.is_dragging_3d_view = False

        self.init_ui()

    def init_ui(self):
        # 主窗口布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # 控制面板
        self.create_control_panel()

        # 显示区域
        self.create_display_area()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请确保目录下存在 marker_params.json")

    def create_control_panel(self):
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        # 标定参数
        self.btn_load_calib = QPushButton("加载标定参数")
        self.btn_load_calib.clicked.connect(self.load_calibration)

        # 帧处理
        self.btn_first_frame = QPushButton("处理首帧")
        self.btn_first_frame.clicked.connect(self.process_first_frame)
        self.btn_first_frame.setEnabled(False)

        self.btn_next_frame = QPushButton("处理后续帧")
        self.btn_next_frame.clicked.connect(self.process_next_frame)
        self.btn_next_frame.setEnabled(False)

        self.btn_batch_process = QPushButton("批量处理")
        self.btn_batch_process.clicked.connect(self.batch_process)
        self.btn_batch_process.setEnabled(False)

        # 视频处理功能已移除（请将视频导出为图片序列后放入图像文件夹进行处理）

        # 保存结果
        self.btn_save_results = QPushButton("保存结果")
        self.btn_save_results.clicked.connect(self.save_results)
        self.btn_save_results.setEnabled(False)

        # 在保存结果按钮后添加数据后处理按钮
        self.btn_post_process = QPushButton("数据后处理")
        self.btn_post_process.clicked.connect(self.post_process_data)
        self.btn_post_process.setEnabled(True)

        # 参数设置
        self.cb_visualize = QCheckBox("实时可视化")
        self.cb_visualize.setChecked(True)

        self.cb_save_error_frames = QCheckBox("保存出错帧")
        self.cb_save_error_frames.setChecked(False)  # 默认不保存
        self.cb_save_error_frames.setToolTip("批量处理时保存有死点的帧的可视化图像")

        # 添加到布局
        control_layout.addWidget(self.btn_load_calib)
        control_layout.addWidget(self.btn_first_frame)
        control_layout.addWidget(self.btn_next_frame)
        control_layout.addWidget(self.btn_batch_process)
        # control_layout.addWidget(self.btn_load_video)
        # control_layout.addWidget(self.btn_start_video)
        # control_layout.addWidget(self.btn_stop_video)
        control_layout.addWidget(self.btn_save_results)
        control_layout.addWidget(self.btn_post_process)
        control_layout.addWidget(self.cb_visualize)
        control_layout.addWidget(self.cb_save_error_frames)

        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)

        ### 参数标定文件
        # 在控制面板添加标定参数路径输入框
        self.calib_group = QGroupBox("标定参数设置")
        calib_layout = QHBoxLayout()

        # 添加标签
        calib_layout.addWidget(QLabel("标定文件夹路径:"))

        # 添加路径输入框
        self.txt_calib_path = QLineEdit()
        self.txt_calib_path.setPlaceholderText("请输入标定参数文件夹路径")
        self.txt_calib_path.setText("calibration_1mm_12X9_0227_Paras")  # 设置默认值
        self.txt_calib_path.setMinimumWidth(300)
        calib_layout.addWidget(self.txt_calib_path)

        # 添加浏览按钮
        self.btn_browse_calib = QPushButton("浏览...")
        self.btn_browse_calib.clicked.connect(self.browse_calib_folder)
        calib_layout.addWidget(self.btn_browse_calib)

        self.calib_group.setLayout(calib_layout)
        self.main_layout.insertWidget(1, self.calib_group)  # 插入到第二个位置

        ### 【修改部分：图像数据文件夹选择】 ###
        self.image_group = QGroupBox("图像数据设置")
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("图像文件夹路径:"))

        self.txt_image_path = QLineEdit()
        self.txt_image_path.setPlaceholderText("请选择图像所在的文件夹")
        self.txt_image_path.setMinimumWidth(300)
        image_layout.addWidget(self.txt_image_path)

        self.btn_browse_image = QPushButton("浏览...")
        self.btn_browse_image.clicked.connect(self.browse_image_folder)
        image_layout.addWidget(self.btn_browse_image)

        self.image_group.setLayout(image_layout)
        self.main_layout.insertWidget(2, self.image_group)  # 插入到第三个位置
        ### 【修改结束】 ###

    def create_display_area(self):
        self.tabs = QTabWidget()

        # 2D/3D视图标签页
        self.tab_view = QWidget()
        self.tab_view_layout = QHBoxLayout(self.tab_view)

        # 2D视图
        self.fig_2d = plt.figure()
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self)

        # 3D视图
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)

        # 2D视图布局
        vbox_2d = QVBoxLayout()
        vbox_2d.addWidget(self.toolbar_2d)
        vbox_2d.addWidget(self.canvas_2d)

        # 3D视图布局
        vbox_3d = QVBoxLayout()
        vbox_3d.addWidget(self.toolbar_3d)
        vbox_3d.addWidget(self.canvas_3d)

        # 添加到标签页
        self.tab_view_layout.addLayout(vbox_2d)
        self.tab_view_layout.addLayout(vbox_3d)

        # 为3D canvas添加鼠标事件监听，用于保存视角
        self.canvas_3d.mpl_connect('button_release_event', self.on_3d_canvas_mouse_release)
        self.canvas_3d.mpl_connect('button_press_event', self.on_3d_canvas_mouse_press)
        self.canvas_3d.mpl_connect('motion_notify_event', self.on_3d_canvas_mouse_motion)

        # 信息标签页
        self.tab_info = QWidget()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout = QVBoxLayout(self.tab_info)
        info_layout.addWidget(self.info_text)



        # 参数设置标签页
        self.tab_settings = QWidget()
        self.create_settings_tab()

        # 添加标签页
        self.tabs.addTab(self.tab_view, "视图")
        self.tabs.addTab(self.tab_info, "信息")
        # self.tabs.addTab(self.tab_matching, "点匹配")
        self.tabs.addTab(self.tab_settings, "参数设置")

        self.main_layout.addWidget(self.tabs)


    def create_settings_tab(self):
        """
        修改后的设置标签页：
        删除了手动设置检测参数的控件，改为使用 marker_params.json
        """
        layout = QVBoxLayout(self.tab_settings)

        # 检测参数 (已修改)
        detection_group = QGroupBox("标记点检测配置")
        detection_layout = QVBoxLayout()

        info_label = QLabel(
            "注意：标记点检测算法已更新。\n"
            "参数现在自动从 'marker_params.json' 文件加载。\n"
            "如需修改检测参数（阈值、面积、圆度等），\n"
            "请直接编辑该 JSON 文件。"
        )
        info_label.setStyleSheet("color: blue; font-weight: bold;")
        detection_layout.addWidget(info_label)

        detection_group.setLayout(detection_layout)

        # 匹配参数
        matching_group = QGroupBox("点匹配参数")
        matching_layout = QVBoxLayout()

        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 500)
        self.spin_max_dist.setValue(50)
        self.spin_max_dist.setPrefix("最大匹配距离: ")

        matching_layout.addWidget(self.spin_max_dist)
        matching_group.setLayout(matching_layout)

        # 3D重建参数
        recon_group = QGroupBox("3D重建参数")
        recon_layout = QVBoxLayout()

        self.combo_surface_degree = QComboBox()
        self.combo_surface_degree.addItems(["1", "2", "3"])
        self.combo_surface_degree.setCurrentIndex(1)
        self.combo_surface_degree.setCurrentText("2")

        recon_layout.addWidget(QLabel("曲面拟合次数:"))
        recon_layout.addWidget(self.combo_surface_degree)
        recon_group.setLayout(recon_layout)

        layout.addWidget(detection_group)
        layout.addWidget(matching_group)
        layout.addWidget(recon_group)
        layout.addStretch()

    def load_calibration(self):
        options = QFileDialog.Options()
        dir_path = self.txt_calib_path.text().strip()

        if not dir_path:
            QMessageBox.warning(self, "警告", "请输入标定参数文件夹路径")
            return False

        if not os.path.isdir(dir_path):
            QMessageBox.warning(self, "警告", f"指定的路径不是有效文件夹:\n{dir_path}")
            return False

        try:
            self.stereo_params = self.load_stereo_params(dir_path)
            self.validate_params(self.stereo_params)
            self.calib_loaded = True
            self.btn_first_frame.setEnabled(True)

            self.status_bar.showMessage("标定参数已加载")
            self.log_message(f"成功加载标定参数: {dir_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标定参数失败: {str(e)}")

    def browse_calib_folder(self):
        """浏览标定参数文件夹"""
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择标定参数所在文件夹",
            self.txt_calib_path.text(),
            options=options)

        if dir_path:
            self.txt_calib_path.setText(dir_path)

    def browse_image_folder(self):
        """浏览图像文件夹"""
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择图像所在的文件夹",
            self.txt_image_path.text(),
            options=options)

        if dir_path:
            self.txt_image_path.setText(dir_path)

    def validate_params(self, params):
        """验证标定参数有效性"""
        assert params['K1'].shape == (3, 3), "K1必须是3x3矩阵"
        assert params['K2'].shape == (3, 3), "K2必须是3x3矩阵"
        assert params['D1'].shape == (1, 5), "D1应为5参数[k1,k2,p1,p2,k3]"
        assert params['R'].shape == (3, 3), "R必须是3x3旋转矩阵"
        assert params['T'].size == 3, "T应为3维平移向量"

        # 检查旋转矩阵正交性
        I = np.dot(params['R'], params['R'].T)
        if not np.allclose(I, np.eye(3), atol=1e-4):
            self.log_message("警告：旋转矩阵不满足正交性！")

    def process_first_frame(self):
        if not self.calib_loaded:
            QMessageBox.warning(self, "警告", "请先加载标定参数")
            return

        self.ALL_FRAME_POINTS = []

        # 直接从文本框读取路径
        dir_path = self.txt_image_path.text().strip()
        if not dir_path or not os.path.isdir(dir_path):
            QMessageBox.warning(self, "警告", "请先选择有效的图像文件夹")
            return

        # 寻找该文件夹下的第一张图片
        try:
            image_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not image_files:
                QMessageBox.warning(self, "警告", "选定的文件夹中没有图片文件")
                return

            first_image_name = image_files[0]
            file_path = os.path.join(dir_path, first_image_name)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"访问图像文件夹失败: {str(e)}")
            return

        if file_path:
            # 更新工作目录
            self.working_directory = os.path.dirname(file_path)
            try:
                initial_points = self.initialize_first_frame(file_path, self.stereo_params)

                self.ALL_FRAME_POINTS.append({
                    'frame_num': 0,
                    'points': initial_points,
                    'timestamp': os.path.getmtime(file_path),
                    'transform_applied': self.FRAME_DATA['transform_rotation'] is not None,
                    'left_points_2d': self.FRAME_DATA.get('left_points_0_R'),
                    'right_points_2d': self.FRAME_DATA.get('right_points_0_R')
                })

                # 立即更新视图
                if self.cb_visualize.isChecked():
                    left_2d = self.FRAME_DATA.get('left_points_0_R')
                    right_2d = self.FRAME_DATA.get('right_points_0_R')

                    self.update_2d_view_mark1(file_path, left_2d, right_2d)
                    self.update_3d_view(initial_points)
                    QApplication.processEvents()

                self.first_frame_processed = True
                self.current_frame = 0

                self.btn_next_frame.setEnabled(True)
                self.btn_batch_process.setEnabled(True)
                self.btn_save_results.setEnabled(True)

                self.status_bar.showMessage("首帧处理完成")
                self.log_message(f"首帧处理完成: {file_path}\n检测到{len(initial_points)}个标记点")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "错误", f"处理首帧失败: {str(e)}")

    def read_image(self, path):
        """安全的读取图片方法，支持中文路径。"""
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取图片失败: {path}, 错误: {e}")
            return None

    def initialize_first_frame(self, image_path, calib_data):
        """处理并初始化首帧数据"""

        # 加载标定参数
        K1 = calib_data['K1']
        D1 = calib_data['D1']
        K2 = calib_data['K2']
        D2 = calib_data['D2']
        R = calib_data['R']
        T = calib_data['T']

        # 处理首帧
        img = self.read_image(image_path)
        if img is None:
            raise ValueError(f"无法加载图片: {image_path}")

        h, w = img.shape[:2]
        mirror_axis = w // 2

        # 生成镜像视图
        left_img = np.full((h, w, 3), 255, dtype=np.uint8)
        left_img[:, :mirror_axis] = img[:, :mirror_axis]
        right_img = np.full((h, w, 3), 255, dtype=np.uint8)
        right_img[:, mirror_axis:] = img[:, mirror_axis:]

        # ----------------- ROI/Match 自动路径加载逻辑 -----------------
        # 获取图像所在目录
        image_dir = os.path.dirname(image_path)
        # 构造data目录路径
        data_dir = os.path.join(image_dir, "data")
        # 确保data目录存在
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                self.log_message(f"已自动创建数据文件夹: {data_dir}")
            except Exception as e:
                print(f"创建data目录失败: {e}")

        # 自动定义文件名
        roi_file = os.path.join(data_dir, "roi_masks.npz")
        history_file = os.path.join(data_dir, "matched_points.npz")

        roi_loaded = False

        if os.path.exists(roi_file):
            try:
                with np.load(roi_file) as data:
                    if 'left_mask' not in data or 'right_mask' not in data:
                        raise KeyError("缺少 mask 数据")
                    left_mask = data['left_mask']
                    right_mask = data['right_mask']
                    if left_mask.ndim == 3: left_mask = left_mask[:, :, 0]
                    if right_mask.ndim == 3: right_mask = right_mask[:, :, 0]
                    self.drawing['left']['mask'] = left_mask
                    self.drawing['right']['mask'] = right_mask
                    self.log_message("✓ 已加载ROI区域")
                    roi_loaded = True
            except Exception as e:
                self.log_message(f"⚠ ROI加载失败: {e}")

        # 如果没有文件或加载失败，要求用户预先准备 ROI 文件（本程序不支持交互式绘制）
        if not roi_loaded:
            QMessageBox.warning(self, "缺少ROI", f"未找到 ROI 文件: {roi_file}\n请在图像目录的 data 子文件夹中准备 'roi_masks.npz' 后重试。")
            return []


        # 检测标记点 (使用新算法)
        detector = self.create_detector()
        raw_left = detector.detect(left_img)
        raw_right = detector.detect(right_img)

        left_circles = self.apply_roi_mask(raw_left, self.drawing['left']['mask'])
        right_circles = self.apply_roi_mask(raw_right, self.drawing['right']['mask'])

        # 立体校正
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        left_points = np.array([(x, y) for (x, y, _) in left_circles], dtype=np.float32)
        right_points = np.array([(x, y) for (x, y, _) in right_circles], dtype=np.float32)
        left_pts = cv2.undistortPoints(left_points, K1, D1, R=R1, P=P1).squeeze() if len(left_points) > 0 else np.array([])
        right_pts = cv2.undistortPoints(right_points, K2, D2, R=R2, P=P2).squeeze() if len(right_points) > 0 else np.array([])

        # 历史匹配数据加载
        matched_pairs = []
        load_success = False

        if os.path.exists(history_file):
            try:
                with np.load(history_file) as data:
                    # 简化处理，假设数据格式正确
                    # 注意：如果加载的历史点数与当前检测点数不一致，可能会导致匹配错误
                    # 这里假设用户是在当前检测结果基础上做的匹配
                    matched_pairs = data['matched_pairs'].tolist()
                    mirror_axis = int(data['mirror_axis'])
                    load_success = True
                    self.log_message("✓ 已加载历史匹配数据")
            except Exception as e:
                self.log_message(f"⚠ 匹配数据加载失败: {e}")

        if not load_success:
            if len(left_circles) == 0 or len(right_circles) == 0:
                QMessageBox.warning(self, "错误", f"检测点数不足 (L:{len(left_circles)}, R:{len(right_circles)})")
                return []
            QMessageBox.warning(self, "警告", "匹配文件不存在，请先确保已生成 data/matched_points.npz")
            return []

        # 3D重建
        if len(matched_pairs) > 0 and len(left_pts) > 0 and len(right_pts) > 0:
            pts1 = left_pts[[i for i, j in matched_pairs]]
            pts2 = right_pts[[j for i, j in matched_pairs]]
            points_3d = self.linear_triangulation(pts1, pts2, P1, P2)

            pts1_R = left_points[[i for i, j in matched_pairs]]
            pts2_R = right_points[[j for i, j in matched_pairs]]
        else:
             return []

        # 更新全局数据
        self.FRAME_DATA.update({
            'initialized': True,
            'roi_masks': {'left': self.drawing['left']['mask'], 'right': self.drawing['right']['mask']},
            'P1': P1, 'P2': P2,
            'left_points_0': pts1, 'right_points_0': pts2,
            'left_points_0_R': pts1_R, 'right_points_0_R': pts2_R,
            'left_points_0_pre': pts1_R, 'right_points_0_pre': pts2_R,
            'base_3d_points': points_3d, 'mirror_axis': mirror_axis,
            'transform_origin': None, 'transform_rotation': None
        })

        # 坐标系构建 - 使用PCA自动确定主方向
        if len(points_3d) >= 3:
            self.log_message("开始构建局部坐标系...")

            try:
                # 使用PCA方法构建坐标系（自动适应点云分布）
                origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)

                self.FRAME_DATA['transform_origin'] = origin
                self.FRAME_DATA['transform_rotation'] = rotation_matrix
                self.log_message(f"✓ 局部坐标系构建成功 (PCA自动拟合)")

            except Exception as e:
                self.log_message(f"⚠ 构建局部坐标系失败: {e}，将使用世界坐标系")
        else:
            self.log_message(f"⚠ 检测点数量({len(points_3d)})过少，无法构建3D坐标系")

        # 3D可视化更新
        if self.FRAME_DATA['transform_rotation'] is not None:
            local_points = self.transform_to_local_coordinates(points_3d)
            self.FRAME_DATA['ref_surface'] = self.PolynomialSurface(local_points, degree=2)
            if self.cb_visualize.isChecked():
                self.plot_reference_surface(local_points, self.FRAME_DATA['ref_surface'])
                self.plot_3d_points(local_points)
        else:
            if self.cb_visualize.isChecked():
                self.plot_3d_points(points_3d)

        self.frame_validity = [1]

        # 计算相邻点距离（仅在首帧时输出）
        if len(pts1_R) > 1:
            dist = np.linalg.norm(pts1_R[0] - pts1_R[1])
            self.log_message(f"标记点平均间距: {dist:.1f} pixels")

        return points_3d

    def process_next_frame(self):
        if not self.first_frame_processed:
            QMessageBox.warning(self, "警告", "请先处理首帧")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择后续帧图像", "", "图像文件 (*.jpg *.png)", options=options)

        if file_path:
            try:
                self.working_directory = os.path.dirname(file_path)
                # 接收 5 个返回值
                points_3d, left_pts, right_pts, left_lost, right_lost = self.process_frame_2D3D(file_path, self.stereo_params)

                # 更新显示
                if self.cb_visualize.isChecked():
                    self.update_2d_view_mark1(file_path, left_pts, right_pts, left_lost, right_lost)
                    self.update_3d_view(points_3d)

                self.current_frame += 1
                self.status_bar.showMessage(f"第{self.current_frame}帧处理完成")
                self.log_message(f"帧{self.current_frame}处理完成: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"处理帧失败: {str(e)}")

    def process_frame(self, image_path, calib_data):
        """处理后续帧 (视频/单帧通用)"""
        if not self.FRAME_DATA['initialized']:
            raise RuntimeError("必须先初始化首帧数据")

        # 加载标定参数
        K1 = calib_data['K1']
        D1 = calib_data['D1']
        K2 = calib_data['K2']
        D2 = calib_data['D2']
        R = calib_data['R']
        T = calib_data['T']

        # 加载参数 (使用已存储的投影矩阵)
        P1 = self.FRAME_DATA['P1']
        P2 = self.FRAME_DATA['P2']
        mirror_axis = self.FRAME_DATA['mirror_axis']

        # 加载图像
        img = self.read_image(image_path)
        if img is None:
            raise ValueError(f"无法加载图片: {image_path}")
        h, w = img.shape[:2]

        # 生成镜像视图
        left_img = np.full((h, w, 3), 255, dtype=np.uint8)
        left_img[:, :mirror_axis] = img[:, :mirror_axis]
        right_img = np.full((h, w, 3), 255, dtype=np.uint8)
        right_img[:, mirror_axis:] = img[:, mirror_axis:]

        # 检测标记点 (使用新算法)
        detector = self.create_detector()
        raw_left = detector.detect(left_img)
        raw_right = detector.detect(right_img)

        left_circles = self.apply_roi_mask(raw_left, self.drawing['left']['mask'])
        right_circles = self.apply_roi_mask(raw_right, self.drawing['right']['mask'])

        left_points = np.array([(x, y) for (x, y, _) in left_circles], dtype=np.float32)
        right_points = np.array([(x, y) for (x, y, _) in right_circles], dtype=np.float32)

        # 去畸变校正 (使用已存储的R1, R2, P1, P2)
        # 需要重新计算R1, R2以保持一致性
        R1, R2, _, _, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        left_pts = cv2.undistortPoints(left_points, K1, D1, R=R1, P=P1).squeeze() if len(left_points) > 0 else np.array([])
        right_pts = cv2.undistortPoints(right_points, K2, D2, R=R2, P=P2).squeeze() if len(right_points) > 0 else np.array([])

        # 自动匹配
        max_dist = self.spin_max_dist.value()
        matched_pairs = self.auto_match_points(
            left_points,
            right_points,
            self.FRAME_DATA['left_points_0_pre'],
            self.FRAME_DATA['right_points_0_pre'],
            max_dist=max_dist
        )

        # 三维重建
        pts1 = left_pts[[i for i, j in matched_pairs if i != 100 and j != 100]]
        pts2 = right_pts[[j for i, j in matched_pairs if i != 100 and j != 100]]

        if len(pts1) > 0:
            points_3d = self.linear_triangulation(pts1, pts2, P1, P2)
        else:
            points_3d = np.array([])

        # 获取匹配后的2D点用于下一帧参考
        matched_left_2d = np.array([left_points[i] if i != 100 else self.FRAME_DATA['left_points_0_pre'][idx]
                                     for idx, (i, j) in enumerate(matched_pairs)], dtype=np.float32)
        matched_right_2d = np.array([right_points[j] if j != 100 else self.FRAME_DATA['right_points_0_pre'][idx]
                                      for idx, (i, j) in enumerate(matched_pairs)], dtype=np.float32)

        # 更新前一帧参考点
        self.FRAME_DATA['left_points_0_pre'] = matched_left_2d
        self.FRAME_DATA['right_points_0_pre'] = matched_right_2d

        # 转换到局部坐标系
        if self.FRAME_DATA['transform_rotation'] is not None and len(points_3d) > 0:
            local_points = self.transform_to_local_coordinates(points_3d)
            # 计算形变量
            deformation = self.calculate_deformation(local_points)

            self.ALL_FRAME_POINTS.append({
                'frame_num': self.current_frame,
                'points': points_3d,
                'timestamp': os.path.getmtime(image_path),
                'transform_applied': True,
                'deformation': deformation,
                'left_points_2d': matched_left_2d,
                'right_points_2d': matched_right_2d
            })
        else:
            self.ALL_FRAME_POINTS.append({
                'frame_num': self.current_frame,
                'points': points_3d,
                'timestamp': os.path.getmtime(image_path),
                'transform_applied': False,
                'left_points_2d': matched_left_2d,
                'right_points_2d': matched_right_2d
            })

        return points_3d

    def process_frame_2D3D(self, image_path, calib_data):
        """处理后续帧：使用【局部近邻位移补偿】算法推算跟丢点的坐标，并记录丢失状态"""
        if not self.FRAME_DATA['initialized']:
            raise RuntimeError("必须先初始化首帧数据")

        # --- 1. 基础参数与检测 ---
        K1 = calib_data['K1']; D1 = calib_data['D1']
        K2 = calib_data['K2']; D2 = calib_data['D2']
        R = calib_data['R']; T = calib_data['T']
        P1 = self.FRAME_DATA['P1']; P2 = self.FRAME_DATA['P2']
        mirror_axis = self.FRAME_DATA['mirror_axis']

        img = self.read_image(image_path)
        if img is None: return [], [], [], []
        h, w = img.shape[:2]

        left_img = np.full((h, w, 3), 255, dtype=np.uint8)
        left_img[:, :mirror_axis] = img[:, :mirror_axis]
        right_img = np.full((h, w, 3), 255, dtype=np.uint8)
        right_img[:, mirror_axis:] = img[:, mirror_axis:]

        detector = self.create_detector()
        left_pts_det = np.array([(x, y) for (x, y, _) in self.apply_roi_mask(detector.detect(left_img), self.drawing['left']['mask'])], dtype=np.float32)
        right_pts_det = np.array([(x, y) for (x, y, _) in self.apply_roi_mask(detector.detect(right_img), self.drawing['right']['mask'])], dtype=np.float32)

        # --- 2. 双向锁定匹配 ---
        matched_pairs = self.auto_match_points(
            left_pts_det, right_pts_det,
            self.FRAME_DATA['left_points_0_pre'], self.FRAME_DATA['right_points_0_pre'],
            max_dist=self.spin_max_dist.value()
        )

        # --- 3. 局部近邻运动补偿逻辑 ---
        found_indices = [idx for idx, (i, j) in enumerate(matched_pairs) if i != 100 and j != 100]
        lost_indices = [idx for idx, (i, j) in enumerate(matched_pairs) if i == 100 or j == 100]

        pre_l = self.FRAME_DATA['left_points_0_pre']
        pre_r = self.FRAME_DATA['right_points_0_pre']

        left_points_R = np.zeros_like(pre_l)
        right_points_R = np.zeros_like(pre_r)

        # 新增：创建两个掩码，分别标记左右两侧哪些点是丢失后预测的
        left_lost_mask = np.zeros(len(pre_l), dtype=bool)
        right_lost_mask = np.zeros(len(pre_r), dtype=bool)

        # A. 先填入已找到的点
        found_diffs_l = []
        found_diffs_r = []
        for idx in found_indices:
            det_i, det_j = matched_pairs[idx]
            left_points_R[idx] = left_pts_det[det_i]
            right_points_R[idx] = right_pts_det[det_j]
            found_diffs_l.append(left_points_R[idx] - pre_l[idx])
            found_diffs_r.append(right_points_R[idx] - pre_r[idx])

        # B. 为丢失的点寻找局部近邻并补偿位移
        if len(found_indices) > 0:
            found_diffs_l = np.array(found_diffs_l)
            found_diffs_r = np.array(found_diffs_r)
            found_pre_l = pre_l[found_indices]
            found_pre_r = pre_r[found_indices]

            for idx in lost_indices:
                det_i, det_j = matched_pairs[idx]

                # 左侧处理
                if det_i == 100:
                    # 左侧丢失，需要预测
                    left_lost_mask[idx] = True
                    dists_l = np.linalg.norm(found_pre_l - pre_l[idx], axis=1)
                    k = min(3, len(found_indices))
                    near_idx_l = np.argsort(dists_l)[:k]
                    local_move_l = np.mean(found_diffs_l[near_idx_l], axis=0)
                    left_points_R[idx] = pre_l[idx] + local_move_l
                else:
                    # 左侧检测到了，直接使用
                    left_lost_mask[idx] = False
                    left_points_R[idx] = left_pts_det[det_i]

                # 右侧处理
                if det_j == 100:
                    # 右侧丢失，需要预测
                    right_lost_mask[idx] = True
                    dists_r = np.linalg.norm(found_pre_r - pre_r[idx], axis=1)
                    k = min(3, len(found_indices))
                    near_idx_r = np.argsort(dists_r)[:k]
                    local_move_r = np.mean(found_diffs_r[near_idx_r], axis=0)
                    right_points_R[idx] = pre_r[idx] + local_move_r
                else:
                    # 右侧检测到了，直接使用
                    right_lost_mask[idx] = False
                    right_points_R[idx] = right_pts_det[det_j]
        else:
            left_points_R = pre_l.copy()
            right_points_R = pre_r.copy()
            left_lost_mask[:] = True
            right_lost_mask[:] = True

        # --- 4. 3D重建与保存 ---
        left_points_R = left_points_R.astype(np.float32)
        right_points_R = right_points_R.astype(np.float32)
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(calib_data['K1'], calib_data['D1'], calib_data['K2'], calib_data['D2'], (w, h), calib_data['R'], calib_data['T'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        l_pts_ud = cv2.undistortPoints(left_points_R, calib_data['K1'], calib_data['D1'], R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_points_R, calib_data['K2'], calib_data['D2'], R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

        self.ALL_FRAME_POINTS.append({
            'frame_num': self.current_frame,
            'points': points_3d,
            'timestamp': os.path.getmtime(image_path),
            'left_points_2d': left_points_R,
            'right_points_2d': right_points_R,
            'left_lost_mask': left_lost_mask,
            'right_lost_mask': right_lost_mask,
            'transform_applied': self.FRAME_DATA['transform_rotation'] is not None
        })

        return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask

    def batch_process(self):
        """批量处理：严格更新上一帧参考点，并检测出错帧"""
        if not self.first_frame_processed:
            QMessageBox.warning(self, "警告", "请先处理首帧")
            return

        dir_path = self.txt_image_path.text().strip()
        image_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))])

        # 只有在勾选了"保存出错帧"时才创建 error 文件夹
        error_dir = None
        if self.cb_save_error_frames.isChecked():
            error_dir = os.path.join(dir_path, "error")
            if os.path.exists(error_dir):
                shutil.rmtree(error_dir)
                self.log_message(f"✓ 已清空旧的 error 文件夹")
            os.makedirs(error_dir)
            self.log_message(f"✓ 已创建 error 文件夹: {error_dir}")
        else:
            self.log_message("✓ 未勾选'保存出错帧'，跳过出错帧保存")

        skipped_frames = 0  # 统计跳过的帧数
        for i, img_file in enumerate(image_files[1:], start=1):
            self.current_frame = i
            file_path = os.path.join(dir_path, img_file)

            # 处理当前帧，接收 5 个返回值
            points_3d, left_2d, right_2d, left_lost, right_lost = self.process_frame_2D3D(file_path, self.stereo_params)

            # 计算无效点数量（左侧或右侧丢失的点）
            total_points = len(left_lost) if left_lost is not None else 0
            invalid_count = np.sum(left_lost | right_lost) if total_points > 0 else 0

            # 如果无效点达到总数的1/3，跳过这一帧
            if total_points > 0 and invalid_count >= total_points / 3:
                self.log_message(f"⚠ 帧 {i} 无效点过多 ({invalid_count}/{total_points})，已跳过")
                # 从 ALL_FRAME_POINTS 中移除最后添加的数据
                if self.ALL_FRAME_POINTS and self.ALL_FRAME_POINTS[-1]['frame_num'] == self.current_frame:
                    self.ALL_FRAME_POINTS.pop()
                skipped_frames += 1
                # 不更新参考点，继续下一帧
                continue

            # 检测出错帧：只在勾选了"保存出错帧"时才保存
            if self.cb_save_error_frames.isChecked() and (np.any(left_lost) or np.any(right_lost)):
                self.save_error_frame_visualization(file_path, left_2d, right_2d, left_lost, right_lost, error_dir)

            # 更新参考坐标
            self.FRAME_DATA['left_points_0_pre'] = left_2d
            self.FRAME_DATA['right_points_0_pre'] = right_2d

            if self.cb_visualize.isChecked():
                self.update_2d_view_mark1(file_path, left_2d, right_2d, left_lost, right_lost)
                self.update_3d_view(points_3d)
                QApplication.processEvents()

            self.status_bar.showMessage(f"处理中: {i}/{len(image_files)-1}")

        # 批量处理完成后输出统计信息
        self.log_message(f"✓ 批量处理完成，共跳过 {skipped_frames} 帧（无效点>=1/3）")

    # 视频处理相关方法已移除。
    # 如果需要处理视频，请先将视频导出为连续图片帧并放置在图像文件夹中，然后使用批量处理功能。
    def save_results(self):
        if not self.first_frame_processed:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        if self.working_directory:
            dir_path = os.path.join(self.working_directory, "result")
        else:
            options = QFileDialog.Options()
            dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", options=options)

        if dir_path:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                os.makedirs(dir_path)

                points = self.FRAME_DATA['base_3d_points']
                left_points_0 = self.FRAME_DATA['left_points_0_R']
                right_points_0 = self.FRAME_DATA['right_points_0_R']

                filename = os.path.join(dir_path, f'frame_{0:03d}_points.txt')
                combined_data = np.hstack((points, left_points_0, right_points_0))
                header = 'X(mm)\tY(mm)\tZ(mm)\tLeft_x\tLeft_y\tRight_x\tRight_y'

                np.savetxt(filename, combined_data, fmt='%.3f', delimiter='\t', header=header, comments='')

                for frame_data in self.ALL_FRAME_POINTS:
                    frame_num = frame_data['frame_num']
                    points = frame_data['points']
                    left_points = frame_data['left_points_2d']
                    right_points = frame_data['right_points_2d']

                    filename = os.path.join(dir_path, f'frame_{frame_num:03d}_points.txt')
                    combined_data = np.hstack((points, left_points, right_points))
                    np.savetxt(filename, combined_data, fmt='%.3f', delimiter='\t', header=header, comments='')

                validity_file = os.path.join(dir_path, 'frame_validity.txt')
                np.savetxt(validity_file, np.array(self.frame_validity, dtype=int), fmt='%d', header='Frame validity (1=valid, 0=invalid)', comments='')

                valid_frames = sum(self.frame_validity)
                total_frames = len(self.frame_validity)
                validity_stats = f"\n有效性统计: {valid_frames}/{total_frames} 帧有效 ({valid_frames / total_frames:.1%})"

                self.status_bar.showMessage("结果已保存")
                self.log_message(f"结果已保存到: {dir_path}{validity_stats}")
                QMessageBox.information(self, "成功", f"结果保存完成\n{validity_stats}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")

    def update_2d_view(self, image_path):
        img = self.read_image(image_path)  # 使用支持中文路径的方法
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        ax.imshow(img_rgb)
        ax.axis('off')
        self.fig_2d.suptitle(f"Current frame: {os.path.basename(image_path)}", fontsize=10)
        self.canvas_2d.draw()

    def update_2d_view_mark1(self, image_path, left_points=None, right_points=None, left_lost_mask=None, right_lost_mask=None):
        """更新2D视图并在图像上绘制标记点（分别标记左右两侧丢失状态）"""
        img = self.read_image(image_path)  # 使用支持中文路径的方法
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = img_rgb.copy()

        # 定义颜色 (RGB)
        COLOR_LEFT_OK = (255, 0, 0)      # 正常左：红色
        COLOR_RIGHT_OK = (0, 255, 0)     # 正常右：绿色
        COLOR_LEFT_LOST = (255, 165, 0)  # 左侧丢失：橙色
        COLOR_RIGHT_LOST = (255, 255, 0) # 右侧丢失：黄色

        # 绘制左图像标记点
        if left_points is not None:
            for i, pt in enumerate(left_points):
                x, y = int(pt[0]), int(pt[1])
                # 如果左侧点丢失，使用橙色；否则用红色
                color = COLOR_LEFT_LOST if (left_lost_mask is not None and left_lost_mask[i]) else COLOR_LEFT_OK
                cv2.circle(display_img, (x, y), 4, color, -1)

        # 绘制右图像标记点
        if right_points is not None:
            for i, pt in enumerate(right_points):
                x, y = int(pt[0]), int(pt[1])
                # 如果右侧点丢失，使用黄色；否则用绿色
                color = COLOR_RIGHT_LOST if (right_lost_mask is not None and right_lost_mask[i]) else COLOR_RIGHT_OK
                cv2.circle(display_img, (x, y), 4, color, -1)

        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        ax.imshow(display_img)
        ax.axis('off')

        # 统计丢失点数
        left_lost_count = np.sum(left_lost_mask) if left_lost_mask is not None else 0
        right_lost_count = np.sum(right_lost_mask) if right_lost_mask is not None else 0
        total_points = len(left_points) if left_points is not None else 0

        title_extra = f"(Det: {total_points - left_lost_count - right_lost_count}, L_Lost: {left_lost_count}, R_Lost: {right_lost_count})" if left_points is not None else ""
        self.fig_2d.suptitle(f"Frame: {os.path.basename(image_path)} {title_extra}", fontsize=10)
        self.canvas_2d.draw()

    def update_3d_view(self, points_3d):
        """更新3D视图，使用用户保存的视角或自动计算最佳视角"""
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)
            title = "Local coordinates 3D view"

            # 显示形变云图
            if self.ALL_FRAME_POINTS and 'deformation' in self.ALL_FRAME_POINTS[-1]:
                deformation = self.ALL_FRAME_POINTS[-1]['deformation']
                sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=deformation['values'], cmap='jet', s=50, vmin=0, vmax=1.5)
                cbar = self.fig_3d.colorbar(sc, ax=ax, shrink=0.8)
                cbar.set_label('deformation (mm)', rotation=270, labelpad=15)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)
        else:
            points = points_3d
            title = "World coordinates 3D view"
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        # 设置坐标轴比例
        ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])

        # 使用保存的视角或自动计算最佳视角
        if self.pointcloud_view_saved:
            # 使用用户手动调整并保存的视角
            try:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim, roll=self.pointcloud_view_roll)
            except TypeError:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim)
        else:
            # 自动调整视角：让平面正对观察者
            elev, azim = self.calculate_optimal_view_angle(points)
            ax.view_init(elev=elev, azim=azim)

        self.canvas_3d.draw()

    def log_message(self, message):
        self.info_text.append(message)
        self.info_text.ensureCursorVisible()

    def on_3d_canvas_mouse_press(self, event):
        """
        3D canvas鼠标按下事件处理函数
        当用户开始拖动3D视图时，暂停定时器刷新以防止视角被重置
        """
        self.is_dragging_3d_view = True
        # 如果在后处理播放模式且正在播放，暂时暂停刷新
        if hasattr(self, 'play_timer') and self.play_timer is not None and self.play_timer.isActive():
            self.play_timer.stop()
            self._was_playing_before_drag = True
        else:
            self._was_playing_before_drag = False

    def on_3d_canvas_mouse_motion(self, event):
        """
        3D canvas鼠标移动事件处理函数
        在拖动过程中实时更新保存的视角值
        """
        if not self.is_dragging_3d_view:
            return

        # 获取当前3D axes
        if len(self.fig_3d.axes) == 0:
            return

        ax = self.fig_3d.axes[0]

        # 检查是否是3D axes
        if not hasattr(ax, 'elev') or not hasattr(ax, 'azim'):
            return

        # 实时更新视角值（在拖动过程中持续更新）
        # 同时保存 elev, azim 和 roll（如果存在）
        current_elev = ax.elev
        current_azim = ax.azim
        current_roll = getattr(ax, 'roll', 0)  # matplotlib 3.6+ 支持 roll

        # 根据当前模式保存到对应的变量
        if hasattr(self, 'play_timer') and self.play_timer is not None:
            self.surface_view_elev = current_elev
            self.surface_view_azim = current_azim
            self.surface_view_roll = current_roll
            self.surface_view_saved = True
        elif self.first_frame_processed:
            self.pointcloud_view_elev = current_elev
            self.pointcloud_view_azim = current_azim
            self.pointcloud_view_roll = current_roll
            self.pointcloud_view_saved = True

    def on_3d_canvas_mouse_release(self, event):
        """
        3D canvas鼠标释放事件处理函数
        当用户拖动3D视图后松开鼠标时，确认保存当前视角并恢复播放
        """
        if not self.is_dragging_3d_view:
            return

        self.is_dragging_3d_view = False

        # 获取当前3D axes - 最终确认视角
        if len(self.fig_3d.axes) > 0:
            ax = self.fig_3d.axes[0]
            if hasattr(ax, 'elev') and hasattr(ax, 'azim'):
                current_elev = ax.elev
                current_azim = ax.azim
                current_roll = getattr(ax, 'roll', 0)

                # 判断当前处于哪种模式并保存相应的视角
                if hasattr(self, 'play_timer') and self.play_timer is not None:
                    self.surface_view_elev = current_elev
                    self.surface_view_azim = current_azim
                    self.surface_view_roll = current_roll
                    self.surface_view_saved = True
                    self.log_message(f"✓ 已保存三维表面视角: elev={current_elev:.2f}°, azim={current_azim:.2f}°, roll={current_roll:.2f}°")
                elif self.first_frame_processed:
                    self.pointcloud_view_elev = current_elev
                    self.pointcloud_view_azim = current_azim
                    self.pointcloud_view_roll = current_roll
                    self.pointcloud_view_saved = True
                    self.log_message(f"✓ 已保存点云视角: elev={current_elev:.2f}°, azim={current_azim:.2f}°, roll={current_roll:.2f}°")

        # 如果之前是播放状态，恢复播放
        if hasattr(self, '_was_playing_before_drag') and self._was_playing_before_drag:
            if hasattr(self, 'play_timer') and self.play_timer is not None:
                self.play_timer.start(33)
            self._was_playing_before_drag = False

    # -------------------- 核心算法函数 (已更新为 Code 1 的逻辑) --------------------
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

    def create_detector(self, force_reload=False):
        """
        创建/获取检测器实例（单例模式，避免重复加载配置）
        force_reload: 强制重新加载配置文件
        """
        # 如果已有缓存且不强制重载，直接返回
        if self.detector is not None and not force_reload:
            return self.detector

        # 1. 尝试从界面上的图像路径文本框获取目录
        dir_path = self.txt_image_path.text().strip()

        # 2. 如果文本框为空（例如直接加载视频模式），尝试使用当前工作目录
        if not dir_path and self.working_directory:
            dir_path = self.working_directory

        # 3. 构建配置文件的完整路径
        config_path = "marker_params.json" # 默认回退：当前运行目录

        if dir_path and os.path.isdir(dir_path):
            potential_path = os.path.join(dir_path, "marker_params.json")
            config_path = potential_path

        # 实例化检测器，只在首次或强制重载时打印信息
        verbose = (self.detector is None) or force_reload
        self.detector = self.CircleDetector(config_path=config_path, verbose=verbose)
        return self.detector



    def apply_roi_mask(self, points, mask):
        """应用ROI掩膜过滤检测点"""
        valid_points = []
        if mask is None:
            return points

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        height, width = mask.shape[:2]

        for (x, y, r) in points:
            x_int = int(round(x))
            y_int = int(round(y))
            if 0 <= x_int < width and 0 <= y_int < height:
                if mask[y_int, x_int] > 0:
                    valid_points.append((x_int, y_int, r))
        return valid_points




    def linear_triangulation(self, pts1, pts2, P1, P2):
        num_points = pts1.shape[0]
        points_3d = np.zeros((num_points, 3))
        for i in range(num_points):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            A = np.array([
                x1 * P1[2, :] - P1[0, :],
                y1 * P1[2, :] - P1[1, :],
                x2 * P2[2, :] - P2[0, :],
                y2 * P2[2, :] - P2[1, :]
            ])
            _, _, V = np.linalg.svd(A)
            X_homo = V[-1, :]
            X = X_homo[:3] / X_homo[3]
            points_3d[i] = X
        return points_3d

    def auto_match_points(self, left_points, right_points, left_pre, right_pre, max_dist=100):
        """自动匹配点 - 使用全局匈牙利算法"""
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        left_ref = np.array(left_pre)
        right_ref = np.array(right_pre)
        num_markers = len(left_ref)

        def get_global_hungarian_match(ref_pts, det_pts, threshold):
            """使用全局匈牙利算法进行匹配"""
            if len(ref_pts) == 0 or len(det_pts) == 0:
                return {}

            # 计算距离矩阵
            dist_matrix = cdist(ref_pts, det_pts)

            # 创建代价矩阵，超过阈值的设为大值（避免匹配）
            large_value = threshold * 10
            cost_matrix = dist_matrix.copy()
            cost_matrix[cost_matrix > threshold] = large_value

            # 处理参考点数量与检测点数量不等的情况
            n_ref = len(ref_pts)
            n_det = len(det_pts)

            if n_ref > n_det:
                padding = np.full((n_ref, n_ref - n_det), large_value)
                cost_matrix = np.hstack([cost_matrix, padding])
            elif n_det > n_ref:
                padding = np.full((n_det - n_ref, n_det), large_value)
                cost_matrix = np.vstack([cost_matrix, padding])

            # 使用匈牙利算法求解全局最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 构建匹配结果，过滤掉超过阈值的匹配
            final_map = {}
            for r_idx, d_idx in zip(row_ind, col_ind):
                if r_idx < n_ref and d_idx < n_det:
                    if dist_matrix[r_idx, d_idx] <= threshold:
                        final_map[r_idx] = d_idx

            return final_map

        # 分别计算左右视图映射
        l_map = get_global_hungarian_match(left_ref, left_points, max_dist)
        r_map = get_global_hungarian_match(right_ref, right_points, max_dist)

        # 整合为系统需要的 (left_idx, right_idx) 格式
        matched = []
        frame_ok = 1
        for i in range(num_markers):
            l_idx = l_map.get(i, 100)
            r_idx = r_map.get(i, 100)
            # 修正：保留左右两侧的独立状态，而不是一旦有一侧丢失就全部设为100
            matched.append((l_idx, r_idx))
            if l_idx == 100 or r_idx == 100:
                frame_ok = 0

        # 更新帧有效性
        if len(self.frame_validity) <= self.current_frame:
            self.frame_validity.append(1)
        self.frame_validity[self.current_frame] = frame_ok

        return matched

    def save_error_frame_visualization(self, image_path, left_points, right_points, left_lost_mask, right_lost_mask, error_dir):
        """
        保存出错帧的可视化图像（增强版：清晰区分左死/右死/双死）

        参数:
        - image_path: 原始图像路径
        - left_points: 左侧点坐标（包含预测点）
        - right_points: 右侧点坐标（包含预测点）
        - left_lost_mask: 左侧丢失掩码（True=丢失，False=检测到）
        - right_lost_mask: 右侧丢失掩码（True=丢失，False=检测到）
        - error_dir: 保存目录
        """
        # 读取原始图像
        img = self.read_image(image_path)  # 使用支持中文路径的方法
        if img is None:
            return

        # 获取镜像轴位置和上一帧参考点
        mirror_axis = self.FRAME_DATA.get('mirror_axis')
        left_pre = self.FRAME_DATA.get('left_points_0_pre')
        right_pre = self.FRAME_DATA.get('right_points_0_pre')

        if mirror_axis is None or left_pre is None or right_pre is None:
            return

        # 创建可视化图像
        vis_img = img.copy()

        # 定义颜色（BGR格式）
        COLOR_NORMAL = (0, 255, 0)      # 绿色：正常检测的配对点
        COLOR_LEFT_LOST = (0, 0, 255)   # 红色：左侧未检测到
        COLOR_RIGHT_LOST = (255, 0, 0)  # 蓝色：右侧未检测到
        COLOR_BOTH_LOST = (0, 165, 255) # 橙色：两侧都未检测到
        COLOR_LINE = (255, 255, 255)    # 白色：连线

        # 统计各类型数量
        both_ok = 0
        left_lost_only = 0
        right_lost_only = 0
        both_lost = 0

        # 绘制配对点和连线
        for i in range(len(left_points)):
            l_pt = left_points[i]
            r_pt = right_points[i]

            x_l, y_l = int(l_pt[0]), int(l_pt[1])
            x_r, y_r = int(r_pt[0]), int(r_pt[1])

            left_is_lost = left_lost_mask[i]
            right_is_lost = right_lost_mask[i]

            # 判断该配对点的状态
            if not left_is_lost and not right_is_lost:
                # 两侧都检测到：绿色（不绘制连线）
                color = COLOR_NORMAL
                label = f"{i}:OK"
                both_ok += 1
                # 只绘制点，不绘制连线
                cv2.circle(vis_img, (x_l, y_l), 5, color, -1)
                cv2.circle(vis_img, (x_r, y_r), 5, color, -1)

            elif left_is_lost and right_is_lost:
                # 两侧都未检测到：橙色
                color = COLOR_BOTH_LOST
                label = f"{i}:L+R"
                both_lost += 1
                # 绘制虚线连接上一帧位置
                x_l_pre, y_l_pre = int(left_pre[i][0]), int(left_pre[i][1])
                x_r_pre, y_r_pre = int(right_pre[i][0]), int(right_pre[i][1])
                # 绘制从上一帧到当前帧的箭头
                cv2.arrowedLine(vis_img, (x_l_pre, y_l_pre), (x_l, y_l), color, 2, tipLength=0.3)
                cv2.arrowedLine(vis_img, (x_r_pre, y_r_pre), (x_r, y_r), color, 2, tipLength=0.3)
                # 绘制预测点（空心圆）
                cv2.circle(vis_img, (x_l, y_l), 6, color, 2)
                cv2.circle(vis_img, (x_r, y_r), 6, color, 2)
                # 绘制跨镜像的连线
                cv2.line(vis_img, (x_l, y_l), (x_r, y_r), color, 1, cv2.LINE_AA)

            elif left_is_lost:
                # 只有左侧未检测到：红色
                color = COLOR_LEFT_LOST
                label = f"{i}:L"
                left_lost_only += 1
                # 左侧绘制预测箭头
                x_l_pre, y_l_pre = int(left_pre[i][0]), int(left_pre[i][1])
                cv2.arrowedLine(vis_img, (x_l_pre, y_l_pre), (x_l, y_l), color, 2, tipLength=0.3)
                cv2.circle(vis_img, (x_l, y_l), 6, color, 2)
                # 右侧正常点
                cv2.circle(vis_img, (x_r, y_r), 5, COLOR_NORMAL, -1)
                # 连接线
                cv2.line(vis_img, (x_l, y_l), (x_r, y_r), color, 1, cv2.LINE_AA)

            else:  # right_is_lost
                # 只有右侧未检测到：蓝色
                color = COLOR_RIGHT_LOST
                label = f"{i}:R"
                right_lost_only += 1
                # 左侧正常点
                cv2.circle(vis_img, (x_l, y_l), 5, COLOR_NORMAL, -1)
                # 右侧绘制预测箭头
                x_r_pre, y_r_pre = int(right_pre[i][0]), int(right_pre[i][1])
                cv2.arrowedLine(vis_img, (x_r_pre, y_r_pre), (x_r, y_r), color, 2, tipLength=0.3)
                cv2.circle(vis_img, (x_r, y_r), 6, color, 2)
                # 连接线
                cv2.line(vis_img, (x_l, y_l), (x_r, y_r), color, 1, cv2.LINE_AA)

            # 添加点编号标签（仅对出错点）
            if left_is_lost or right_is_lost:
                # 在点的位置绘制标签
                label_y = max(10, y_l - 10)
                cv2.putText(vis_img, label, (x_l - 20, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制镜像分割线
        cv2.line(vis_img, (mirror_axis, 0), (mirror_axis, img.shape[0]), (128, 128, 128), 2)

        # 添加图例和统计信息
        legend_y = 30
        line_height = 30
        font_scale = 0.7
        thickness = 2

        cv2.putText(vis_img, f"Frame {self.current_frame} Error Analysis:",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        legend_y += line_height
        cv2.putText(vis_img, f"Normal (Green): {both_ok}",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_NORMAL, thickness)

        legend_y += line_height
        cv2.putText(vis_img, f"Left Lost (Red): {left_lost_only}",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_LEFT_LOST, thickness)

        legend_y += line_height
        cv2.putText(vis_img, f"Right Lost (Blue): {right_lost_only}",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_RIGHT_LOST, thickness)

        legend_y += line_height
        cv2.putText(vis_img, f"Both Lost (Orange): {both_lost}",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BOTH_LOST, thickness)

        # 保存图像
        frame_name = os.path.basename(image_path)
        save_path = os.path.join(error_dir, f"error_frame_{self.current_frame:03d}_{frame_name}")
        cv2.imwrite(save_path, vis_img)

        # 日志输出
        self.log_message(
            f"⚠ 出错帧 {self.current_frame}: "
            f"L单死:{left_lost_only}, R单死:{right_lost_only}, 双死:{both_lost} "
            f"-> 已保存至 {os.path.basename(save_path)}"
        )

    def build_coordinate_system_pca(self, points_3d):
        """
        使用PCA自动构建坐标系（自动适应点云分布）

        原理：
        1. 以点云质心为原点
        2. 使用PCA找到点云的主方向作为坐标轴
        3. 第一主成分作为X轴（最大变化方向）
        4. 第二主成分作为Y轴（次大变化方向）
        5. 第三主成分作为Z轴（法向量方向，对于平面点云这是最小变化方向）

        优点：对于平面点云，Z轴自动垂直于平面
        """
        # 计算质心作为原点
        centroid = np.mean(points_3d, axis=0)

        # 中心化点云
        centered = points_3d - centroid

        # PCA分析
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值从大到小排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 构建旋转矩阵（主成分作为新坐标轴）
        x_axis = eigenvectors[:, 0]  # 第一主成分（最大变化方向）
        y_axis = eigenvectors[:, 1]  # 第二主成分（次大变化方向）
        z_axis = eigenvectors[:, 2]  # 第三主成分（最小变化方向，即法向量）

        # 确保Z轴指向正方向（法向量向上）
        if z_axis[2] < 0:
            z_axis = -z_axis

        # 重新构建右手坐标系
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

        return centroid, rotation_matrix

    def transform_to_local_coordinates(self, points):
        """
        使用PCA自动构建坐标系（自动适应点云分布）

        原理：
        1. 以点云质心为原点
        2. 使用PCA找到点云的主方向作为坐标轴
        3. 第一主成分作为X轴（最大变化方向）
        4. 第二主成分作为Y轴（次大变化方向）
        5. 第三主成分作为Z轴（法向量方向，对于平面点云这是最小变化方向）

        优点：对于平面点云，Z轴自动垂直于平面
        """
        origin = self.FRAME_DATA['transform_origin']
        rotation = self.FRAME_DATA['transform_rotation']
        translated = points - origin
        return np.dot(translated, rotation.T)

    def calculate_optimal_view_angle(self, points):
        """
        计算最佳视角，让平面点云正对观察者

        对于平面点云：
        - Z轴是法向量方向（垂直于平面）
        - 需要从正上方往下看，才能看清整个平面
        """
        # 从正上方往下看（俯视图）
        elev = 90  # 俯仰角90度，从正上方垂直向下看
        azim = 0   # 方位角0度

        return elev, azim

    def load_stereo_params(self, param_dir):
        params = {}
        params['K1'] = np.loadtxt(f"{param_dir}/K1.txt").reshape(3, 3).astype(np.float64)
        params['K2'] = np.loadtxt(f"{param_dir}/K2.txt").reshape(3, 3).astype(np.float64)
        params['D1'] = np.loadtxt(f"{param_dir}/D1.txt").reshape(1, 5).astype(np.float64)
        params['D2'] = np.loadtxt(f"{param_dir}/D2.txt").reshape(1, 5).astype(np.float64)
        params['R'] = np.loadtxt(f"{param_dir}/R.txt").reshape(3, 3).astype(np.float64)
        params['T'] = np.loadtxt(f"{param_dir}/T.txt").reshape(3, 1).astype(np.float64)
        return params

    class PolynomialSurface:
        def __init__(self, points, degree=2):
            self.degree = degree
            self.coeffs = self._fit_polynomial(points)

        def _fit_polynomial(self, points):
            x = points[:, 0]; y = points[:, 1]; z = points[:, 2]
            terms = []
            for d in range(self.degree + 1):
                for i in range(d + 1):
                    terms.append(x ** (d - i) * y ** i)
            A = np.vstack(terms).T
            coeffs, residuals, _, _ = np.linalg.lstsq(A, z, rcond=None)
            return coeffs

        def evaluate(self, x, y):
            terms = []
            for d in range(self.degree + 1):
                for i in range(d + 1):
                    terms.append(x ** (d - i) * y ** i)
            return np.dot(np.vstack(terms).T, self.coeffs)

    def calculate_deformation(self, local_points):
        deformation = np.zeros(len(local_points))
        basic_points_3d = self.FRAME_DATA['base_3d_points']
        basic_local_points = self.transform_to_local_coordinates(basic_points_3d)
        for i, (x, y, z) in enumerate(local_points):
            deformation[i] = np.linalg.norm(local_points[i] - basic_local_points[i])
        deformation = deformation - np.min(deformation)
        return {'values': deformation, 'max': np.max(np.abs(deformation)), 'min': np.min(deformation), 'mean': np.mean(deformation)}

    def plot_3d_points(self, points_3d, title="3D Reconstruction"):
        """绘制3D点云，使用用户保存的视角或自动计算最佳视角"""
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                           c='b', cmap='viridis', s=50, edgecolor='k', depthshade=True)

        ax.set_box_aspect([np.ptp(points_3d[:, 0]), np.ptp(points_3d[:, 1]), np.ptp(points_3d[:, 2])])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, pad=20)

        # 使用保存的视角或自动计算最佳视角
        if self.pointcloud_view_saved:
            try:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim, roll=self.pointcloud_view_roll)
            except TypeError:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim)
        else:
            elev, azim = self.calculate_optimal_view_angle(points_3d)
            ax.view_init(elev=elev, azim=azim)

        self.canvas_3d.draw()

    def plot_reference_surface(self, points, surface, title=None):
        """绘制参考曲面，使用用户保存的视角或自动计算最佳视角"""
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        # 绘制参考点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='b', s=50, edgecolor='k', label='Reference point')

        # 生成拟合曲面网格
        x = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
        y = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
        X, Y = np.meshgrid(x, y)
        Z = surface.evaluate(X.flatten(), Y.flatten()).reshape(X.shape)

        # 绘制曲面
        ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])
        surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis',
                              antialiased=True, label='Fitting surface')

        cbar = self.fig_3d.colorbar(surf, ax=ax, shrink=0.8)
        cbar.set_label('Height (mm)', rotation=270, labelpad=15)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        if title is None:
            title = f"Poly fitting surface (degree {surface.degree})"
        ax.set_title(title, pad=20)

        # 使用保存的视角或自动计算最佳视角
        if self.pointcloud_view_saved:
            try:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim, roll=self.pointcloud_view_roll)
            except TypeError:
                ax.view_init(elev=self.pointcloud_view_elev, azim=self.pointcloud_view_azim)
        else:
            elev, azim = self.calculate_optimal_view_angle(points)
            ax.view_init(elev=elev, azim=azim)

        self.canvas_3d.draw()

    def post_process_data(self):
        """数据后处理主函数：自动读取路径"""
        base_dir = self.txt_image_path.text().strip()
        valid_path = False
        if base_dir and os.path.isdir(base_dir):
            potential_result_dir = os.path.join(base_dir, "result")
            if os.path.isdir(potential_result_dir):
                self.image_dir = base_dir
                self.data_dir = potential_result_dir
                valid_path = True
                self.log_message(f"自动加载后处理路径:\n图像: {self.image_dir}\n数据: {self.data_dir}")
            else:
                QMessageBox.warning(self, "路径错误", f"在图像目录下未找到 'result' 文件夹:\n{base_dir}")
                return
        else:
            QMessageBox.warning(self, "路径错误", "请先在'图像数据设置'中填入有效的图像文件夹路径")
            return

        try:
            self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('frame_') and f.endswith('_points.txt')], key=lambda x: int(x.split('_')[1]))
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=lambda x: os.path.splitext(x)[0])
            if len(self.data_files) == 0:
                QMessageBox.warning(self, "警告", "result文件夹中没有找到数据文件")
                return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取文件列表失败: {str(e)}")
            return

        self.frame_validity = []
        validity_file = os.path.join(self.data_dir, 'frame_validity.txt')
        if os.path.exists(validity_file):
            try:
                self.frame_validity = np.loadtxt(validity_file, dtype=int).tolist()
            except Exception as e:
                self.frame_validity = [1] * len(self.data_files)
        else:
            self.frame_validity = [1] * len(self.data_files)

        self.post_process_data_list = []
        self.reference_points = None
        try:
            data_file = self.data_files[0]
            data_path = os.path.join(self.data_dir, data_file)
            data = np.genfromtxt(open(data_path, 'r'), skip_header=1)
            if data.ndim == 1: data = data.reshape(1, -1)
            if data.shape[1] >= 3:
                self.reference_points = data[:, :3].copy()
            else:
                raise ValueError("数据文件格式错误，列数不足")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")
            return

        self.current_play_frame = 0
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.update_post_process_display)
        self.create_playback_controls()
        self.play_timer.start(33)
        self.status_bar.showMessage("数据后处理播放中...")

    def update_post_process_display(self):
        """更新后处理显示"""
        data_file = self.data_files[self.current_play_frame]
        img_file = self.image_files[self.current_play_frame]
        data_path = os.path.join(self.data_dir, data_file)
        img_path = os.path.join(self.image_dir, img_file)

        try:
            data = np.genfromtxt(open(data_path, 'r'), skip_header=1)
            if data.shape[1] < 7: self.log_message(f"数据文件 {data_file} 列数不足")
            points_3d = data[:, :3]
            left_points = data[:, 3:5]
            right_points = data[:, 5:7]
            displacements = np.linalg.norm(points_3d - self.reference_points, axis=1)
        except Exception as e:
            self.log_message(f"加载数据文件 {data_file} 失败: {str(e)}")

        try:
            is_valid = self.frame_validity[self.current_play_frame] if self.current_play_frame < len(self.frame_validity) else 1
            if is_valid:
                self.update_2d_view_mark1_R(img_path, left_points, right_points)
                self.update_3d_view_post_map(points_3d, displacements)
        except Exception as e:
            self.log_message(f"显示帧 {self.current_play_frame} 失败: {str(e)}")

        self.lbl_frame.setText(f"帧号: {self.current_play_frame}")
        self.playback_slider.setValue(self.current_play_frame)
        self.current_play_frame += 1
        if self.current_play_frame >= len(self.data_files):
            self.current_play_frame = 0

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play_pause.setText("播放")
        else:
            self.play_timer.start(33)
            self.btn_play_pause.setText("暂停")

    def stop_playback(self):
        if hasattr(self, 'play_timer') and self.play_timer is not None:
            self.play_timer.stop()
            self.play_timer.deleteLater()
            self.play_timer = None  # 清除引用，让on_3d_canvas_mouse_release知道已退出后处理模式
        if hasattr(self, 'playback_panel'):
            self.main_layout.removeWidget(self.playback_panel)
            self.playback_panel.deleteLater()
            del self.playback_panel
        # 清除首帧数据缓存，以便下次后处理时重新加载
        if hasattr(self, 'first_frame_data'):
            del self.first_frame_data
        self.status_bar.showMessage("播放已停止")

    def set_playback_frame(self, frame_idx):
        frame_idx = max(0, min(frame_idx, len(self.data_files) - 1))
        self.current_play_frame = frame_idx
        if not self.play_timer.isActive():
            self.update_post_process_display() # 手动更新一帧，但不要自增
            self.current_play_frame = frame_idx # update里自增了，这里重置回当前

    def create_playback_controls(self):
        if hasattr(self, 'playback_panel'):
            self.main_layout.removeWidget(self.playback_panel)
            self.playback_panel.deleteLater()

        self.playback_panel = QGroupBox("播放控制")
        playback_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("暂停")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.lbl_frame = QLabel("帧号: 0")
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setRange(0, len(self.data_files) - 1)
        self.playback_slider.valueChanged.connect(self.set_playback_frame)

        playback_layout.addWidget(self.btn_play_pause)
        playback_layout.addWidget(self.btn_stop)
        playback_layout.addWidget(self.lbl_frame)
        playback_layout.addWidget(self.playback_slider)
        self.playback_panel.setLayout(playback_layout)
        self.main_layout.insertWidget(self.main_layout.count() - 1, self.playback_panel)

    def update_2d_view_mark1_R(self, image_path, left_points=None, right_points=None):
        """更新2D视图并在图像上绘制相对偏移箭头"""
        try:
            img = self.read_image(image_path)  # 使用支持中文路径的方法
            if img is None: raise ValueError("无法加载图像")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display_img = img_rgb.copy()
            first_frame_data = self.get_first_frame_data()

            if left_points is not None and len(left_points) > 0 and first_frame_data['left_points'] is not None:
                for pt_current, pt_ref in zip(left_points, first_frame_data['left_points']):
                    x_current, y_current = int(pt_current[0]), int(pt_current[1])
                    x_ref, y_ref = int(pt_ref[0]), int(pt_ref[1])
                    cv2.arrowedLine(display_img, (x_ref, y_ref), (x_current, y_current), (255, 0, 0), 2, tipLength=0.3)
                    cv2.circle(display_img, (x_current, y_current), 5, (255, 0, 0), -1)

            if right_points is not None and len(right_points) > 0 and first_frame_data['right_points'] is not None:
                for pt_current, pt_ref in zip(right_points, first_frame_data['right_points']):
                    x_current, y_current = int(pt_current[0]), int(pt_current[1])
                    x_ref, y_ref = int(pt_ref[0]), int(pt_ref[1])
                    cv2.arrowedLine(display_img, (x_ref, y_ref), (x_current, y_current), (0, 255, 0), 2, tipLength=0.3)
                    cv2.circle(display_img, (x_current, y_current), 5, (0, 255, 0), -1)

            self.fig_2d.clear()
            ax = self.fig_2d.add_subplot(111)
            ax.imshow(display_img)
            ax.axis('off')
            img_title = os.path.splitext(os.path.basename(image_path))[0]
            self.fig_2d.suptitle(f"{img_title} (Frame {self.current_play_frame})", fontsize=10)
            self.canvas_2d.draw()
        except Exception as e:
            self.log_message(f"更新2D视图失败: {str(e)}")

    def get_first_frame_data(self):
        if not hasattr(self, 'first_frame_data'):
            first_data_file = self.data_files[0]
            data_path = os.path.join(self.data_dir, first_data_file)
            try:
                data = np.genfromtxt(open(data_path, 'r'), skip_header=1)
                self.first_frame_data = {'left_points': data[:, 3:5], 'right_points': data[:, 5:7]}
            except Exception as e:
                self.first_frame_data = {'left_points': None, 'right_points': None}
        return self.first_frame_data

    def update_3d_view_post_map(self, points_3d, displacements=None, grid_resolution=50):
        """数据后处理的3D视图更新，使用用户保存的视角或自动计算最佳视角 - 只显示点覆盖的区域"""
        # 如果用户正在拖动视图，跳过刷新以防止视角被重置
        if self.is_dragging_3d_view:
            return

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('jet')

        if displacements is not None and len(points_3d) >= 4:
            x = points_3d[:, 0]
            y = points_3d[:, 1]
            z = points_3d[:, 2]

            try:
                # 使用 Delaunay 三角剖分确定点覆盖的区域
                points_2d = np.column_stack([x, y])
                tri = Delaunay(points_2d)

                # 生成细密网格用于 RBF 插值
                xi = np.linspace(min(x), max(x), grid_resolution)
                yi = np.linspace(min(y), max(y), grid_resolution)
                xi_grid, yi_grid = np.meshgrid(xi, yi)

                # 创建网格点坐标
                grid_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])

                # 检查哪些网格点在 Delaunay 三角形内
                simplex_indices = tri.find_simplex(grid_points)
                valid_mask = simplex_indices >= 0

                # 只保留在三角形内的网格点
                valid_grid_points = grid_points[valid_mask]

                if len(valid_grid_points) > 0:
                    # 对有效的网格点进行 RBF 插值
                    rbf_z = Rbf(x, y, z, function='thin_plate')
                    rbf_disp = Rbf(x, y, displacements, function='thin_plate')

                    zi = rbf_z(valid_grid_points[:, 0], valid_grid_points[:, 1])
                    disp_i = rbf_disp(valid_grid_points[:, 0], valid_grid_points[:, 1])

                    # 绘制曲面（使用散点而不是 plot_surface，避免长方形）
                    norm = plt.Normalize(vmin=0, vmax=1.5)
                    colors = cmap(norm(disp_i))

                    ax.scatter(valid_grid_points[:, 0], valid_grid_points[:, 1], zi,
                              c=colors, s=20, edgecolor='none', depthshade=True, alpha=0.7)

                    cbar = self.fig_3d.colorbar(
                        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, shrink=0.8
                    )
                    cbar.set_label('Displacement (mm)', rotation=270, labelpad=15)
                else:
                    # 如果没有有效点，直接显示原始点
                    sc = ax.scatter(x, y, z, c=displacements, cmap=cmap, s=50,
                                   edgecolor='k', depthshade=True)
                    cbar = self.fig_3d.colorbar(sc, ax=ax, shrink=0.8)
                    cbar.set_label('Displacement (mm)', rotation=270, labelpad=15)

            except Exception as e:
                # 降级处理：直接显示点云
                self.log_message(f"⚠ 处理失败: {str(e)}，使用点云显示")
                sc = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                               c=displacements, cmap=cmap, s=50, edgecolor='k', depthshade=True)
                cbar = self.fig_3d.colorbar(sc, ax=ax, shrink=0.8)
                cbar.set_label('Displacement (mm)', rotation=270, labelpad=15)
        else:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c='b', s=50, edgecolor='k')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([np.ptp(points_3d[:, 0]), np.ptp(points_3d[:, 1]), np.ptp(points_3d[:, 2])])

        # 使用保存的视角或自动计算最佳视角
        if self.surface_view_saved:
            # 使用用户手动调整并保存的视角
            # 检查 matplotlib 版本是否支持 roll 参数
            try:
                ax.view_init(elev=self.surface_view_elev, azim=self.surface_view_azim, roll=self.surface_view_roll)
            except TypeError:
                # 旧版本 matplotlib 不支持 roll 参数
                ax.view_init(elev=self.surface_view_elev, azim=self.surface_view_azim)
        else:
            # 自动调整视角
            elev, azim = self.calculate_optimal_view_angle(points_3d)
            ax.view_init(elev=elev, azim=azim)

        title = "3D Surface with Displacement (Only Coverage Area)"
        ax.set_title(title, pad=20)

        self.canvas_3d.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarkerTrackingSystem()
    window.show()
    sys.exit(app.exec_())