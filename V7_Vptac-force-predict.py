# -*- coding: utf-8 -*-
'''
V7_Vptac-force-predict — 点云实时显示 + 力预测 + 统一采集 + HDF5回放
功能说明：
1. 支持USB摄像头实时采集，逐帧进行标记点检测和三维重建
2. 启动前弹出文件选择窗口：标定文件夹、参数JSON文件、首帧ROI和匹配数据保存文件夹、力预测模型目录
3. GUI中同时显示视频帧和三维点云
4. 六维力传感器实时订阅与显示（坐标轴显示）
5. 力预测模型（V6_force_predict.ForcePredictor）实时推理并显示
6. 统一HDF5采集：
   - /vision (30Hz): timestamp, xyz, dxyz, abnormal, predicted_force
   - /force  (100Hz): timestamp, values (fx,fy,fz,mx,my,mz)
   - /reference: xyz_ref
   - /meta: 实验元信息
7. HDF5数据回放：加载已保存的HDF5文件进行可视化分析
'''
import sys
import signal
import os
_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
os.add_dll_directory(_utils_dir)
sys.path.insert(0, _utils_dir)
import json
import time
import threading
from datetime import datetime
import numpy as np
import h5py
import cv2
import topic  # type: ignore
import message  # type: ignore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QGroupBox, QStatusBar, QSpinBox, QSplitter, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment

from V6_force_predict_lightnet import ForcePredictor

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────── 相机线程 ───────────────────────

class CameraThread(QThread):
    """摄像头采集线程"""
    frame_signal = pyqtSignal(np.ndarray, float)

    def __init__(self, camera_index=0, fps=30, rotate_180=True):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = fps
        self.rotate_180 = rotate_180
        self.running = True
        self.video_cap = None

    def run(self):
        self.video_cap = cv2.VideoCapture(self.camera_index)
        if not self.video_cap.isOpened():
            return
        self.video_cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        frame_interval = 1.0 / self.target_fps

        while self.running:
            start_time = time.time()
            ret, frame = self.video_cap.read()
            if ret:
                capture_timestamp = time.time()
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.frame_signal.emit(frame, capture_timestamp)
            elapsed = time.time() - start_time
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        if self.video_cap:
            self.video_cap.release()

    def stop(self):
        self.running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


# ─────────────────────── 帧处理线程 ───────────────────────

class FrameProcessThread(QThread):
    """帧处理线程 - 将耗时的检测和重建放到后台"""
    result_signal = pyqtSignal(object, object, object, object, object, object, object, object)

    def __init__(self):
        super().__init__()
        self.running = True
        self.frame_queue = []
        self.lock = threading.Lock()
        self.process_func = None
        self.has_new_frame = threading.Event()

    def set_processor(self, process_func):
        self.process_func = process_func

    def add_frame(self, frame, timestamp=0.0):
        with self.lock:
            self.frame_queue = [(frame, timestamp)]
        self.has_new_frame.set()

    def run(self):
        while self.running:
            self.has_new_frame.wait(timeout=0.1)
            if not self.running:
                break
            frame = None
            timestamp = 0.0
            with self.lock:
                if self.frame_queue:
                    frame, timestamp = self.frame_queue.pop(0)
            self.has_new_frame.clear()

            if frame is not None and self.process_func is not None:
                try:
                    points_3d, left_pts, right_pts, left_lost, right_lost, is_abnormal = self.process_func(frame)
                    self.result_signal.emit(frame, points_3d, left_pts, right_pts,
                                            left_lost, right_lost, timestamp, is_abnormal)
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


# ─────────────────────── 圆检测器 ───────────────────────

class CircleDetector:
    def __init__(self, config_path="marker_params.json", verbose=True):
        self.params = {
            "blur": 3, "block_size": 11, "c_val": 5, "morph_size": 2,
            "min_area": 30, "max_area": 1000, "circularity": 70, "inertia": 30
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_params = json.load(f)
                    self.params.update(saved_params)
                if verbose:
                    print(f"已加载参数配置: {config_path}")
            except Exception as e:
                if verbose:
                    print(f"⚠ 加载参数文件失败，使用默认值: {e}")
        else:
            if verbose:
                print("⚠ 未找到参数文件，使用系统默认参数")

    def _detect_by_local_minima(self, gray, blur):
        """局部极小值检测方法"""
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


# ─────────────────────── 主窗口 ───────────────────────

class V7MainWindow(QMainWindow):
    """V7 主窗口: 点云显示 + 力预测 + 统一 HDF5 采集"""

    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_input_config_v7.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("V7 - 点云实时显示 & 力预测")
        self.setGeometry(50, 50, 1600, 1000)

        # 路径变量
        self.calib_dir = None
        self.params_json_path = None
        self.data_save_dir = None
        self.model_dir = None  # 力预测模型目录

        # 摄像头相关
        self.camera_thread = None
        self.camera_index = 0
        self.rotate_180 = True
        self.latest_camera_frame = None

        # 帧处理线程
        self.process_thread = None
        self.last_3d_update_time = 0
        self.min_3d_update_interval = 0.1

        # 性能优化：点云刷新控制
        self.pointcloud_frame_counter = 0
        self.pointcloud_update_interval = 3  # 每3帧更新一次点云显示（10Hz）

        # 标定参数
        self.stereo_params = None
        self.calib_loaded = False

        # 帧数据
        self.FRAME_DATA = {
            'initialized': False,
            'roi_masks': None,
            'P1': None, 'P2': None,
            'left_points_0': None, 'right_points_0': None,
            'left_points_0_R': None, 'right_points_0_R': None,
            'left_points_0_pre': None, 'right_points_0_pre': None,
            'base_3d_points': None,
            'mirror_axis': None,
            'transform_origin': None,
            'transform_rotation': None,
            'left_edges': None, 'right_edges': None,
            'current_left_crossings': [], 'current_right_crossings': [],
        }
        self.drawing = {
            'left': {'mask': None},
            'right': {'mask': None}
        }

        # 检测器
        self.detector = None

        # 匹配参数
        self.max_match_dist = 50
        self.consecutive_abnormal_frames = 0

        # 3D 视角
        self.view_elev = 90
        self.view_azim = 0
        self.view_roll = -90
        self.view_saved = False
        self.is_dragging = False
        self._updating_base = False
        self.drag_release_time = 0
        self.drag_cooldown = 0.5
        self._was_playing_before_drag = False

        # 播放控制
        self.is_playing = False
        self.current_frame_idx = 0
        self.fps = 60

        # 位移统计
        self.avg_displacement_vector = np.array([0.0, 0.0, 0.0])
        self.avg_displacement_magnitude = 0.0
        self.max_displacement = 0.0
        self.min_displacement = 0.0

        # 力预测器
        self.force_predictor = None  # type: ForcePredictor | None
        self.latest_predicted_force = None  # (output_dim,) ndarray

        # 六维力传感器
        self.ft_node = None
        self.ft_subscription = None
        self.ft_lock = threading.Lock()
        self.latest_ft_data = None

        # HDF5回放相关
        self.h5_file = None
        self.h5_mode = False
        self.h5_vision_data = None
        self.h5_force_data = None
        self.h5_xyz_ref = None
        self.h5_current_idx = 0
        self.h5_playing = False
        self.latest_ft_timestamp = 0.0
        self.ft_bias = np.zeros(6, dtype=np.float64)  # 力传感器调零偏置
        self.ft_recent_buffer = []  # 最近N个采样，用于自动调零

        # 力数据历史缓冲（用于坐标轴绘图）
        self.force_history_len = 300
        self.force_actual_history = np.zeros((self.force_history_len, 6))
        self.force_pred_history = np.zeros((self.force_history_len, 6))
        self.force_history_idx = 0

        # ── 统一采集缓冲 ──
        self.is_recording = False
        # 视觉帧 (camera rate ~30Hz)
        self.rec_vision = {
            'timestamps': [],
            'xyz': [],
            'abnormal': [],
            'predicted_force': [],
        }
        # 力传感器 (sensor rate ~100Hz)
        self.rec_force = {
            'timestamps': [],
            'ft_values': [],
        }

        # 初始化界面
        self.init_ui()

        # 启动六维力传感器订阅
        self.start_ft_subscription()

        # 自动加载上次配置
        QTimer.singleShot(100, self.auto_load_last_config)

    # ──────────────────── UI ────────────────────

    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        self.btn_select_input = QPushButton("选择输入源")
        self.btn_select_input.clicked.connect(self.select_input)

        self.btn_load_h5 = QPushButton("加载HDF5")
        self.btn_load_h5.clicked.connect(self.load_h5_file)

        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)

        self.btn_reset = QPushButton("重置首帧")
        self.btn_reset.clicked.connect(self.reset_first_frame)
        self.btn_reset.setEnabled(False)

        self.btn_set_as_base = QPushButton("设置为首帧")
        self.btn_set_as_base.clicked.connect(self.set_current_as_base)
        self.btn_set_as_base.setEnabled(False)

        self.btn_record = QPushButton("开始采集")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)

        self.lbl_frame_info = QLabel("帧: 0")

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 120)
        self.spin_speed.setValue(60)
        self.spin_speed.setPrefix("速度: ")
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.valueChanged.connect(self.update_playback_speed)

        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 500)
        self.spin_max_dist.setValue(50)
        self.spin_max_dist.setPrefix("匹配距离: ")
        self.spin_max_dist.valueChanged.connect(lambda v: setattr(self, 'max_match_dist', v))

        self.lbl_camera = QLabel("摄像头:")
        self.spin_camera_idx = QSpinBox()
        self.spin_camera_idx.setRange(0, 10)
        self.spin_camera_idx.setValue(0)
        self.spin_camera_idx.setPrefix("ID: ")

        self.chk_rotate = QCheckBox("旋转180°")
        self.chk_rotate.setChecked(True)

        for w in [self.btn_select_input, self.btn_load_h5, self.btn_play_pause, self.btn_reset,
                  self.btn_set_as_base, self.btn_record, self.lbl_frame_info,
                  self.spin_speed, self.spin_max_dist, self.lbl_camera,
                  self.spin_camera_idx, self.chk_rotate]:
            control_layout.addWidget(w)
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 主显示区域
        main_splitter = QSplitter(Qt.Vertical)

        # 上部：视觉 (视频 + 3D)
        visual_splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.lbl_video = QLabel("视频显示区域")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(640, 400)
        self.lbl_video.setStyleSheet("background-color: #2a2a2a; color: white;")
        left_layout.addWidget(self.lbl_video)
        visual_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.fig_3d = plt.figure(figsize=(5, 4))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
        right_layout.addWidget(self.toolbar_3d)
        right_layout.addWidget(self.canvas_3d)
        visual_splitter.addWidget(right_widget)

        visual_splitter.setSizes([700, 500])
        main_splitter.addWidget(visual_splitter)

        # 下部：力曲线坐标轴
        force_widget = QWidget()
        force_layout = QVBoxLayout(force_widget)
        force_layout.setContentsMargins(0, 0, 0, 0)

        # 调零按钮（紧凑一行）
        btn_layout = QHBoxLayout()
        self.btn_ft_auto_zero = QPushButton("自动调零")
        self.btn_ft_auto_zero.clicked.connect(self.auto_zero_ft)
        self.btn_ft_reset_zero = QPushButton("重置调零")
        self.btn_ft_reset_zero.clicked.connect(self.reset_zero_ft)
        btn_layout.addWidget(self.btn_ft_auto_zero)
        btn_layout.addWidget(self.btn_ft_reset_zero)
        btn_layout.addStretch()
        force_layout.addLayout(btn_layout)

        # 6个力分量坐标轴（红色=实际力, 蓝色=预测力）
        self.fig_force, self.axes_force = plt.subplots(2, 3, figsize=(15, 6))
        self.canvas_force = FigureCanvas(self.fig_force)
        force_layout.addWidget(self.canvas_force)

        main_splitter.addWidget(force_widget)
        main_splitter.setSizes([500, 500])  # 上下各半，力图不被压扁
        main_layout.addWidget(main_splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择输入源")

        # 3D 视图鼠标事件
        self.canvas_3d.mpl_connect('button_press_event', self.on_3d_mouse_press)
        self.canvas_3d.mpl_connect('button_release_event', self.on_3d_mouse_release)
        self.canvas_3d.mpl_connect('motion_notify_event', self.on_3d_mouse_motion)

        # 定时器：matplotlib 力曲线刷新 (blit模式，200ms = 5Hz)
        self.force_plot_timer = QTimer(self)
        self.force_plot_timer.timeout.connect(self.update_force_plot)
        self.force_plot_timer.start(200)

    # ──────────────────── 六维力传感器 ────────────────────

    def start_ft_subscription(self):
        try:
            ft_options = topic.NodeOptions()
            ft_options.node_name = 'v7_ft_subscriber'
            ft_options.sub_url = 'tcp://192.168.50.1:19091'
            self.ft_node = topic.Node(ft_options)
            if not self.ft_node.Start():
                print("六维力传感器节点启动失败")
                self.ft_node = None
                return
            self.ft_subscription = self.ft_node.CreateSubscriptionRT(
                "system_rtstate", self._on_ft_data)
            print("六维力传感器订阅已启动")
        except Exception as e:
            print(f"六维力传感器订阅启动失败: {e}")
            self.ft_node = None

    def _on_ft_data(self, tt: topic.SystemRtState):
        parm = message.SystemStateData()
        message.display_rt(tt, parm)
        timestamp = time.time()
        with self.ft_lock:
            self.latest_ft_data = parm.controller.ftvalues
            self.latest_ft_timestamp = timestamp
            # 维护最近采样的环形缓冲（用于自动调零）
            if parm.controller.ftvalues:
                for ftv in parm.controller.ftvalues:
                    raw = [ftv.fx, ftv.fy, ftv.fz, ftv.mx, ftv.my, ftv.mz]
                    self.ft_recent_buffer.append(raw)
                    if len(self.ft_recent_buffer) > 100:
                        self.ft_recent_buffer.pop(0)
                    # 采集时保存调零后的数据
                    if self.is_recording:
                        zeroed = [raw[i] - self.ft_bias[i] for i in range(6)]
                        self.rec_force['timestamps'].append(timestamp)
                        self.rec_force['ft_values'].append(zeroed)

    def _init_force_plot(self):
        """首次初始化力曲线子图 + 保存静态背景用于 blit"""
        self._force_lines_actual = []
        self._force_lines_pred = []
        self._force_vlines = []
        self._force_ylims = [(-1.0, 1.0)] * 6  # 每个子图的 y 范围
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        x = np.arange(self.force_history_len)

        for i, ax in enumerate(self.axes_force.flat):
            ax.clear()
            line_actual, = ax.plot(x, np.zeros(self.force_history_len),
                                   'r-', label='实际', linewidth=1.5, alpha=0.7, animated=True)
            line_pred, = ax.plot(x, np.zeros(self.force_history_len),
                                 'b-', label='预测', linewidth=1.5, alpha=0.7, animated=True)
            vline = ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.8, animated=True)
            ax.set_title(labels[i], fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, self.force_history_len])
            ax.set_ylim(-1.0, 1.0)
            self._force_lines_actual.append(line_actual)
            self._force_lines_pred.append(line_pred)
            self._force_vlines.append(vline)

        self.fig_force.tight_layout()
        # 完整绘制一次，然后保存各子图的静态背景（坐标轴/网格/图例/标题）
        self.canvas_force.draw()
        self._force_backgrounds = []
        for ax in self.axes_force.flat:
            self._force_backgrounds.append(self.canvas_force.copy_from_bbox(ax.bbox))
        self._force_plot_ready = True

    def _refresh_force_backgrounds(self, actual=None, pred=None):
        """y 轴范围变化后，更新标题并重绘静态背景"""
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        if actual is not None and pred is not None:
            for i, ax in enumerate(self.axes_force.flat):
                ax.set_title(f'{labels[i]}: 实际={actual[i]:.2f}, 预测={pred[i]:.2f}', fontsize=9)
        self.canvas_force.draw()
        self._force_backgrounds = []
        for ax in self.axes_force.flat:
            self._force_backgrounds.append(self.canvas_force.copy_from_bbox(ax.bbox))

    def update_force_plot(self):
        """力曲线刷新 — 使用 blit 只重绘线条，不重绘整张图"""
        if self.h5_mode:
            return

        # ── 获取当前力数据 ──
        with self.ft_lock:
            ft_data = self.latest_ft_data
            bias = self.ft_bias.copy()

        actual = np.zeros(6)
        if ft_data and len(ft_data) > 0:
            ftv = ft_data[0]
            actual = np.array([ftv.fx - bias[0], ftv.fy - bias[1], ftv.fz - bias[2],
                              ftv.mx - bias[3], ftv.my - bias[4], ftv.mz - bias[5]])

        pred = self.latest_predicted_force if self.latest_predicted_force is not None else np.zeros(6)

        # ── 写入环形缓冲 ──
        self.force_actual_history[self.force_history_idx] = actual
        self.force_pred_history[self.force_history_idx] = pred
        self.force_history_idx = (self.force_history_idx + 1) % self.force_history_len

        # ── 首次初始化 ──
        if not getattr(self, '_force_plot_ready', False):
            self._init_force_plot()
            return

        # ── 检查 y 轴范围是否需要扩展 ──
        needs_bg_refresh = False
        for i in range(6):
            a_col = self.force_actual_history[:, i]
            p_col = self.force_pred_history[:, i]
            data_min = min(a_col.min(), p_col.min())
            data_max = max(a_col.max(), p_col.max())
            lo, hi = self._force_ylims[i]
            if data_min < lo or data_max > hi:
                margin = max(abs(data_max - data_min) * 0.3, 0.5)
                new_lo = data_min - margin
                new_hi = data_max + margin
                self._force_ylims[i] = (new_lo, new_hi)
                self.axes_force.flat[i].set_ylim(new_lo, new_hi)
                needs_bg_refresh = True

        # 每5次(~1秒)刷新一次标题数值，或 y 轴范围变化时也刷新
        if not hasattr(self, '_force_title_counter'):
            self._force_title_counter = 0
        self._force_title_counter += 1
        if self._force_title_counter >= 5 or needs_bg_refresh:
            self._force_title_counter = 0
            needs_bg_refresh = True  # 标题变化也需要重绘背景

        if needs_bg_refresh:
            self._refresh_force_backgrounds(actual, pred)

        # ── blit 更新：只重绘线条 artist ──
        for i, ax in enumerate(self.axes_force.flat):
            self.canvas_force.restore_region(self._force_backgrounds[i])
            self._force_lines_actual[i].set_ydata(self.force_actual_history[:, i])
            self._force_lines_pred[i].set_ydata(self.force_pred_history[:, i])
            self._force_vlines[i].set_xdata([self.force_history_idx, self.force_history_idx])
            ax.draw_artist(self._force_lines_actual[i])
            ax.draw_artist(self._force_lines_pred[i])
            ax.draw_artist(self._force_vlines[i])
            self.canvas_force.blit(ax.bbox)

    # ──────────────────── 力传感器调零 ────────────────────

    def auto_zero_ft(self):
        """自动调零：取最近50个采样的均值作为偏置"""
        with self.ft_lock:
            buf = self.ft_recent_buffer.copy()
        if len(buf) == 0:
            QMessageBox.warning(self, "提示", "尚无力传感器数据，无法调零")
            return
        n = min(50, len(buf))
        recent = np.array(buf[-n:], dtype=np.float64)
        bias = recent.mean(axis=0)
        with self.ft_lock:
            self.ft_bias = bias
        self.status_bar.showMessage(
            f"力传感器已调零（取最近{n}个采样均值）: "
            f"Fx={bias[0]:+.3f} Fy={bias[1]:+.3f} Fz={bias[2]:+.3f} "
            f"Mx={bias[3]:+.3f} My={bias[4]:+.3f} Mz={bias[5]:+.3f}")

    def reset_zero_ft(self):
        """重置调零偏置为零"""
        with self.ft_lock:
            self.ft_bias = np.zeros(6, dtype=np.float64)
        self.status_bar.showMessage("力传感器调零已重置")

    # ──────────────────── 配置加载 / 保存 ────────────────────

    def save_input_config(self):
        config = {
            'calib_dir': self.calib_dir,
            'params_json_path': self.params_json_path,
            'data_save_dir': self.data_save_dir,
            'model_dir': self.model_dir,
            'camera_index': self.spin_camera_idx.value(),
            'rotate_180': self.chk_rotate.isChecked(),
        }
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def load_input_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return False
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            calib_dir = config.get('calib_dir')
            data_save_dir = config.get('data_save_dir')
            model_dir = config.get('model_dir')
            if not calib_dir or not os.path.exists(calib_dir):
                return False
            if not data_save_dir or not os.path.exists(data_save_dir):
                return False
            self.calib_dir = calib_dir
            self.data_save_dir = data_save_dir
            self.params_json_path = config.get('params_json_path')
            self.model_dir = model_dir
            if 'camera_index' in config:
                self.spin_camera_idx.setValue(config['camera_index'])
            if 'rotate_180' in config:
                self.chk_rotate.setChecked(config['rotate_180'])
            return True
        except Exception:
            return False

    def auto_load_last_config(self):
        if not self.load_input_config():
            self.status_bar.showMessage("就绪 - 请选择输入源")
            return
        self.load_and_start()

    # ──────────────────── 输入选择 ────────────────────

    def select_input(self):
        # 停止当前线程
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        self.is_playing = False

        # 1. 标定文件夹
        calib_dir = QFileDialog.getExistingDirectory(
            self, "选择标定参数文件夹（包含K1.txt, K2.txt等）", "")
        if not calib_dir:
            return
        self.calib_dir = calib_dir

        # 2. 参数JSON
        params_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像处理参数文件 (marker_params.json)", "",
            "JSON文件 (*.json);;所有文件 (*.*)")
        self.params_json_path = params_path if params_path else None

        # 3. data 文件夹
        data_dir = QFileDialog.getExistingDirectory(
            self, "选择data文件夹（包含roi_masks.npz）", "")
        if not data_dir:
            return
        self.data_save_dir = data_dir

        # 4. 力预测模型目录（默认指向 data_dir/model_output）
        default_model_dir = os.path.join(data_dir, "model_output")
        model_dir = QFileDialog.getExistingDirectory(
            self, "选择力预测模型目录（包含 model.pth, scaler.npz, train_config.json）",
            default_model_dir if os.path.exists(default_model_dir) else data_dir)
        if not model_dir:
            QMessageBox.warning(self, "提示", "未选择模型目录，力预测功能将不可用")
        self.model_dir = model_dir if model_dir else None

        self.save_input_config()
        self.load_and_start()

    def load_and_start(self):
        """加载所有配置并启动摄像头"""
        if not self.load_calibration():
            return
        self.load_detector()
        if not self.load_reference_data():
            return
        self.load_force_predictor()
        self.start_camera()

    # ──────────────────── 力预测模型加载 ────────────────────

    def load_force_predictor(self):
        if not self.model_dir or not os.path.exists(self.model_dir):
            self.force_predictor = None
            print("力预测模型未加载（目录无效或未指定）")
            return
        try:
            self.force_predictor = ForcePredictor(self.model_dir)
            print(f"力预测模型已加载: {self.model_dir}")
        except Exception as e:
            self.force_predictor = None
            QMessageBox.warning(self, "警告", f"力预测模型加载失败: {e}")

    # ──────────────────── 标定 / 检测器 / 参考数据 ────────────────────

    def load_calibration(self):
        try:
            self.stereo_params = self.load_stereo_params(self.calib_dir)
            self.calib_loaded = True
            self.status_bar.showMessage(f"标定参数已加载: {self.calib_dir}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标定参数失败: {e}")
            return False

    def load_stereo_params(self, param_dir):
        params = {}
        params['K1'] = np.loadtxt(os.path.join(param_dir, "K1.txt")).reshape(3, 3).astype(np.float64)
        params['K2'] = np.loadtxt(os.path.join(param_dir, "K2.txt")).reshape(3, 3).astype(np.float64)
        params['D1'] = np.loadtxt(os.path.join(param_dir, "D1.txt")).reshape(1, 5).astype(np.float64)
        params['D2'] = np.loadtxt(os.path.join(param_dir, "D2.txt")).reshape(1, 5).astype(np.float64)
        params['R'] = np.loadtxt(os.path.join(param_dir, "R.txt")).reshape(3, 3).astype(np.float64)
        params['T'] = np.loadtxt(os.path.join(param_dir, "T.txt")).reshape(3, 1).astype(np.float64)
        return params

    def load_detector(self):
        config_path = self.params_json_path if self.params_json_path else "marker_params.json"
        self.detector = CircleDetector(config_path=config_path, verbose=True)

    def load_reference_data(self):
        roi_file = os.path.join(self.data_save_dir, "roi_masks.npz")
        if not os.path.exists(roi_file):
            QMessageBox.warning(self, "警告", f"未找到ROI文件: {roi_file}")
            return False
        try:
            with np.load(roi_file) as data:
                left_mask = data['left_mask']
                right_mask = data['right_mask']
                if left_mask.ndim == 3:
                    left_mask = left_mask[:, :, 0]
                if right_mask.ndim == 3:
                    right_mask = right_mask[:, :, 0]
                self.drawing['left']['mask'] = left_mask
                self.drawing['right']['mask'] = right_mask
        except Exception as e:
            QMessageBox.warning(self, "警告", f"ROI加载失败: {e}")
            return False

        # mirror_axis
        mirror_axis = 640
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
            except Exception:
                pass

        # 首帧数据
        result_dir = os.path.join(self.data_save_dir, "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")
        if not os.path.exists(first_frame_file):
            QMessageBox.warning(self, "警告", f"未找到首帧数据: {first_frame_file}")
            return False

        try:
            data = np.genfromtxt(first_frame_file, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 7:
                QMessageBox.warning(self, "警告", "首帧数据文件格式错误")
                return False
            points_3d = data[:, :3].astype(np.float64)
            pts1_R = data[:, 3:5].astype(np.float32)
            pts2_R = data[:, 5:7].astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取首帧数据失败: {e}")
            return False

        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']
        h, w = 480, 1280
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        pts1 = cv2.undistortPoints(pts1_R, K1, D1, R=R1, P=P1).squeeze()
        pts2 = cv2.undistortPoints(pts2_R, K2, D2, R=R2, P=P2).squeeze()

        self.FRAME_DATA.update({
            'initialized': True,
            'roi_masks': {'left': self.drawing['left']['mask'], 'right': self.drawing['right']['mask']},
            'P1': P1, 'P2': P2,
            'left_points_0': pts1, 'right_points_0': pts2,
            'left_points_0_R': pts1_R, 'right_points_0_R': pts2_R,
            'left_points_0_pre': pts1_R.copy(), 'right_points_0_pre': pts2_R.copy(),
            'base_3d_points': points_3d,
            'mirror_axis': mirror_axis,
        })

        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix

        # 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        return True

    # ──────────────────── 摄像头启动 ────────────────────

    def start_camera(self):
        self.camera_index = self.spin_camera_idx.value()
        self.rotate_180 = self.chk_rotate.isChecked()

        self.process_thread = FrameProcessThread()
        self.process_thread.set_processor(self.process_frame)
        self.process_thread.result_signal.connect(self.on_process_result)
        self.process_thread.start()

        self.camera_thread = CameraThread(
            camera_index=self.camera_index,
            fps=self.spin_speed.value(),
            rotate_180=self.rotate_180)
        self.camera_thread.frame_signal.connect(self.on_camera_frame)
        self.camera_thread.start()

        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("暂停")
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.is_playing = True
        self.status_bar.showMessage(f"摄像头已启动 (ID: {self.camera_index})")

    def on_camera_frame(self, frame, timestamp=0.0):
        if not self.FRAME_DATA['initialized']:
            return
        self.latest_camera_frame = frame
        self.current_frame_idx += 1

        h, w = frame.shape[:2]
        if self.FRAME_DATA['mirror_axis'] is None or self.FRAME_DATA['mirror_axis'] != w // 2:
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            self.FRAME_DATA['P1'] = P1
            self.FRAME_DATA['P2'] = P2

        if self.current_frame_idx % 10 == 0:
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx}")

        if self.is_playing and self.process_thread is not None:
            self.process_thread.add_frame(frame.copy(), timestamp)
        else:
            self.display_frame(frame)

    def on_process_result(self, frame, points_3d, left_pts, right_pts,
                          left_lost, right_lost, timestamp=0.0, is_abnormal=False):
        if points_3d is None:
            return

        # 力预测（保持每帧推理）
        predicted_force = self._predict_force(points_3d)
        self.latest_predicted_force = predicted_force

        # 采集缓冲
        self.buffer_frame_data(points_3d, timestamp, is_abnormal, predicted_force)

        # 显示
        self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)

        # 点云更新优化：每N帧更新一次（降低刷新率到10Hz）
        self.pointcloud_frame_counter += 1
        if self.pointcloud_frame_counter < self.pointcloud_update_interval:
            return
        self.pointcloud_frame_counter = 0

        if self.is_dragging:
            return
        if time.time() - self.drag_release_time < self.drag_cooldown:
            return
        current_time = time.time()
        if current_time - self.last_3d_update_time >= self.min_3d_update_interval:
            self.update_3d_view(points_3d)
            self.last_3d_update_time = current_time

    # ──────────────────── 力预测 ────────────────────

    def _predict_force(self, points_3d):
        """用 dxyz 做力预测，返回 ndarray 或 None"""
        if self.force_predictor is None:
            return None
        base_3d = self.FRAME_DATA['base_3d_points']
        if base_3d is None or len(points_3d) != len(base_3d):
            return None
        try:
            local_xyz = self.get_local_coords(points_3d)
            base_local = self.get_local_coords(base_3d)
            dxyz = (local_xyz - base_local).astype(np.float32)  # (N, 3)
            return self.force_predictor.predict(dxyz)
        except Exception as e:
            print(f"力预测失败: {e}")
            return None

    # ──────────────────── 采集 (统一 HDF5) ────────────────────

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.FRAME_DATA['initialized']:
            self.status_bar.showMessage("请先初始化首帧")
            return
        self.rec_vision = {'timestamps': [], 'xyz': [], 'abnormal': [], 'predicted_force': []}
        with self.ft_lock:
            self.rec_force = {'timestamps': [], 'ft_values': []}
        self.is_recording = True
        self.btn_record.setText("停止采集")
        self.btn_record.setStyleSheet("background-color: #ff4444; color: white;")
        self.status_bar.showMessage("数据采集中...")

    def stop_recording(self):
        self.is_recording = False
        self.btn_record.setText("开始采集")
        self.btn_record.setStyleSheet("")
        self.save_unified_hdf5()

    def get_local_coords(self, points_3d):
        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)
            points[:, 1] = -points[:, 1]
            return points
        return points_3d.copy()

    def buffer_frame_data(self, points_3d, timestamp, is_abnormal=False, predicted_force=None):
        if not self.is_recording or points_3d is None:
            return
        local_xyz = self.get_local_coords(points_3d)
        self.rec_vision['timestamps'].append(timestamp)
        self.rec_vision['xyz'].append(local_xyz.astype(np.float32))
        self.rec_vision['abnormal'].append(is_abnormal)
        # 预测力：如果为None则填零
        if predicted_force is not None:
            self.rec_vision['predicted_force'].append(predicted_force.astype(np.float32))
        else:
            out_dim = self.force_predictor.config['output_dim'] if self.force_predictor else 6
            self.rec_vision['predicted_force'].append(np.zeros(out_dim, dtype=np.float32))

    def save_unified_hdf5(self):
        """将视觉帧(~30Hz)和力传感器(~100Hz)保存到同一个HDF5文件"""
        stop_time = datetime.now()

        vis_ts = self.rec_vision['timestamps']
        with self.ft_lock:
            ft_ts = self.rec_force['timestamps'].copy()
            ft_vals = self.rec_force['ft_values'].copy()

        if not vis_ts and not ft_ts:
            QMessageBox.warning(self, "警告", "没有采集到任何数据")
            self.status_bar.showMessage("采集已停止（无数据）")
            return

        filename = f"V7_recording_{stop_time.strftime('%Y%m%d_%H%M%S')}.h5"
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        try:
            with h5py.File(filepath, 'w') as f:
                # ── /vision (~30Hz, 与相机帧对齐) ──
                if vis_ts:
                    timestamps_v = np.array(vis_ts, dtype=np.float64)
                    xyz_all = np.array(self.rec_vision['xyz'], dtype=np.float32)     # (T, N, 3)
                    abnormal = np.array(self.rec_vision['abnormal'], dtype=np.bool_)  # (T,)
                    pred_force = np.array(self.rec_vision['predicted_force'], dtype=np.float32)  # (T, out_dim)

                    # dxyz
                    base_3d = self.FRAME_DATA['base_3d_points']
                    xyz_ref = self.get_local_coords(base_3d).astype(np.float32)
                    dxyz = (xyz_all - xyz_ref[np.newaxis, :, :]).astype(np.float32)

                    N = xyz_ref.shape[0]

                    vision = f.create_group('vision')
                    vision.create_dataset('timestamp', data=timestamps_v)
                    vision.create_dataset('xyz', data=xyz_all)
                    vision.create_dataset('dxyz', data=dxyz)
                    vision.create_dataset('abnormal', data=abnormal)
                    vision.create_dataset('predicted_force', data=pred_force)
                    vision.create_dataset('point_id', data=np.arange(N, dtype=np.int32))

                    ref = f.create_group('reference')
                    ref.create_dataset('xyz_ref', data=xyz_ref)

                # ── /force (~100Hz, 力传感器原始采样率) ──
                if ft_ts:
                    timestamps_f = np.array(ft_ts, dtype=np.float64)
                    values_f = np.array(ft_vals, dtype=np.float64)  # (T_ft, 6)

                    force = f.create_group('force')
                    force.create_dataset('timestamp', data=timestamps_f)
                    force.create_dataset('values', data=values_f)
                    force.attrs['columns'] = 'fx,fy,fz,mx,my,mz'

                # ── /meta ──
                meta = f.create_group('meta')
                meta.attrs['camera_fps'] = self.spin_speed.value()
                meta.attrs['marker_count'] = N if vis_ts else 0
                meta.attrs['experiment_name'] = filename
                meta.attrs['vision_frames'] = len(vis_ts)
                meta.attrs['force_samples'] = len(ft_ts)
                meta.attrs['has_force_prediction'] = self.force_predictor is not None
                meta.attrs['ft_bias'] = self.ft_bias.tolist()  # 力传感器调零偏置
                if self.model_dir:
                    meta.attrs['model_dir'] = self.model_dir

            # 清空缓冲
            self.rec_vision = {'timestamps': [], 'xyz': [], 'abnormal': [], 'predicted_force': []}
            with self.ft_lock:
                self.rec_force = {'timestamps': [], 'ft_values': []}

            msg = (f"数据已保存:\n{filepath}\n\n"
                   f"视觉帧: {len(vis_ts)} (~30Hz)\n"
                   f"力传感器采样: {len(ft_ts)} (~100Hz)")
            QMessageBox.information(self, "保存成功", msg)
            self.status_bar.showMessage(f"采集完成: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入HDF5失败:\n{e}")
            self.status_bar.showMessage(f"保存失败: {e}")

    # ──────────────────── 帧处理 (检测+重建) ────────────────────

    def process_frame(self, frame):
        if not self.FRAME_DATA['initialized']:
            return None, None, None, None, None, False

        h, w = frame.shape[:2]
        mirror_axis = self.FRAME_DATA['mirror_axis']

        left_img = np.full((h, w, 3), 255, dtype=np.uint8)
        left_img[:, :mirror_axis] = frame[:, :mirror_axis]
        right_img = np.full((h, w, 3), 255, dtype=np.uint8)
        right_img[:, mirror_axis:] = frame[:, mirror_axis:]

        left_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                     self.apply_roi_mask(self.detector.detect(left_img),
                                                         self.drawing['left']['mask'])], dtype=np.float32)
        right_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                      self.apply_roi_mask(self.detector.detect(right_img),
                                                          self.drawing['right']['mask'])], dtype=np.float32)

        pre_l = self.FRAME_DATA['left_points_0_pre']
        pre_r = self.FRAME_DATA['right_points_0_pre']
        filter_dist = self.max_match_dist * 1.5

        if len(left_pts_det_raw) > 0 and len(pre_l) > 0:
            dist_matrix_l = cdist(left_pts_det_raw, pre_l)
            min_dists_l = np.min(dist_matrix_l, axis=1)
            left_pts_det = left_pts_det_raw[min_dists_l <= filter_dist]
        else:
            left_pts_det = left_pts_det_raw

        if len(right_pts_det_raw) > 0 and len(pre_r) > 0:
            dist_matrix_r = cdist(right_pts_det_raw, pre_r)
            min_dists_r = np.min(dist_matrix_r, axis=1)
            right_pts_det = right_pts_det_raw[min_dists_r <= filter_dist]
        else:
            right_pts_det = right_pts_det_raw

        matched_pairs = self.auto_match_points(
            left_pts_det, right_pts_det,
            self.FRAME_DATA['left_points_0_pre'],
            self.FRAME_DATA['right_points_0_pre'],
            max_dist=self.max_match_dist)

        found_indices = [idx for idx, (i, j) in enumerate(matched_pairs) if i != 100 and j != 100]
        lost_indices = [idx for idx, (i, j) in enumerate(matched_pairs) if i == 100 or j == 100]

        left_points_R = np.zeros_like(pre_l)
        right_points_R = np.zeros_like(pre_r)
        left_lost_mask = np.zeros(len(pre_l), dtype=bool)
        right_lost_mask = np.zeros(len(pre_r), dtype=bool)

        found_diffs_l = []
        found_diffs_r = []
        for idx in found_indices:
            det_i, det_j = matched_pairs[idx]
            left_points_R[idx] = left_pts_det[det_i]
            right_points_R[idx] = right_pts_det[det_j]
            found_diffs_l.append(left_points_R[idx] - pre_l[idx])
            found_diffs_r.append(right_points_R[idx] - pre_r[idx])
        if len(found_indices) > 0:
            found_diffs_l = np.array(found_diffs_l)
            found_diffs_r = np.array(found_diffs_r)
            found_pre_l = pre_l[found_indices]
            found_pre_r = pre_r[found_indices]
            for idx in lost_indices:
                det_i, det_j = matched_pairs[idx]
                if det_i == 100:
                    left_lost_mask[idx] = True
                    dists_l = np.linalg.norm(found_pre_l - pre_l[idx], axis=1)
                    k = min(3, len(found_indices))
                    near_idx_l = np.argsort(dists_l)[:k]
                    local_move_l = np.mean(found_diffs_l[near_idx_l], axis=0)
                    left_points_R[idx] = pre_l[idx] + local_move_l
                else:
                    left_points_R[idx] = left_pts_det[det_i]
                if det_j == 100:
                    right_lost_mask[idx] = True
                    dists_r = np.linalg.norm(found_pre_r - pre_r[idx], axis=1)
                    k = min(3, len(found_indices))
                    near_idx_r = np.argsort(dists_r)[:k]
                    local_move_r = np.mean(found_diffs_r[near_idx_r], axis=0)
                    right_points_R[idx] = pre_r[idx] + local_move_r
                else:
                    right_points_R[idx] = right_pts_det[det_j]
        else:
            left_points_R = pre_l.copy()
            right_points_R = pre_r.copy()
            left_lost_mask[:] = True
            right_lost_mask[:] = True

        # ── 平面图约束 Mesh Untangling ──
        left_edges = self.FRAME_DATA.get('left_edges') or []
        right_edges = self.FRAME_DATA.get('right_edges') or []

        def find_crossing_pairs(pts, edges):
            crossings = []
            n = len(edges)
            if n == 0:
                return crossings
            bboxes = np.empty((n, 4), dtype=np.float64)
            for i, (a, b) in enumerate(edges):
                ax, ay = pts[a][0], pts[a][1]
                bx, by = pts[b][0], pts[b][1]
                bboxes[i, 0] = min(ax, bx)
                bboxes[i, 1] = max(ax, bx)
                bboxes[i, 2] = min(ay, by)
                bboxes[i, 3] = max(ay, by)
            for i in range(n):
                a, b = edges[i]
                p1, p2 = pts[a], pts[b]
                for j in range(i + 1, n):
                    c, d = edges[j]
                    if a == c or a == d or b == c or b == d:
                        continue
                    if (bboxes[i, 1] < bboxes[j, 0] or bboxes[j, 1] < bboxes[i, 0] or
                            bboxes[i, 3] < bboxes[j, 2] or bboxes[j, 3] < bboxes[i, 2]):
                        continue
                    if self.segments_intersect(p1, p2, pts[c], pts[d]):
                        crossings.append((i, j))
            return crossings

        def untangle(pts_R, lost_mask, edges, matched_pairs_ref, side):
            MAX_ITER = 5
            for _ in range(MAX_ITER):
                crossings = find_crossing_pairs(pts_R, edges)
                if not crossings:
                    break
                resolved_any = False
                for ei, ej in crossings:
                    a, b = edges[ei]
                    c, d = edges[ej]
                    candidates = [(a, c), (a, d), (b, c), (b, d)]
                    for p, q in candidates:
                        p_lost = lost_mask[p]
                        q_lost = lost_mask[q]
                        if p_lost or q_lost:
                            for lost_idx in ([p] if p_lost else []) + ([q] if q_lost else []):
                                neighbors = []
                                for ea, eb in edges:
                                    if ea == lost_idx and not lost_mask[eb]:
                                        neighbors.append(eb)
                                    elif eb == lost_idx and not lost_mask[ea]:
                                        neighbors.append(ea)
                                if not neighbors:
                                    continue
                                crossing_nodes = set()
                                for ci, cj in crossings:
                                    for nd in edges[ci] + edges[cj]:
                                        crossing_nodes.add(nd)
                                safe_neighbors = [n for n in neighbors if n not in crossing_nodes]
                                if not safe_neighbors:
                                    continue
                                base_pts = self.FRAME_DATA['left_points_0_R'] if side == 'left' \
                                    else self.FRAME_DATA['right_points_0_R']
                                moves = [pts_R[n] - base_pts[n] for n in safe_neighbors]
                                median_move = np.median(moves, axis=0)
                                safe_pos = base_pts[lost_idx] + median_move
                                pts_R[lost_idx] = 0.5 * pts_R[lost_idx] + 0.5 * safe_pos
                            resolved_any = True
                            break
                        pts_R_trial = pts_R.copy()
                        pts_R_trial[p], pts_R_trial[q] = pts_R[q].copy(), pts_R[p].copy()
                        new_crossings = find_crossing_pairs(pts_R_trial, edges)
                        new_cross_set = set(map(tuple, new_crossings))
                        old_cross_set = set(map(tuple, crossings))
                        if (ei, ej) not in new_cross_set and len(new_cross_set) < len(old_cross_set):
                            pts_R[p], pts_R[q] = pts_R_trial[p].copy(), pts_R_trial[q].copy()
                            resolved_any = True
                            break
                    if resolved_any:
                        break
                if not resolved_any:
                    break

        # ── 灾难性熔断检测 ──
        num_total = len(pre_l)
        num_lost = int(np.sum(left_lost_mask) + np.sum(right_lost_mask))
        lost_ratio = num_lost / (num_total * 2) if num_total > 0 else 1.0
        left_initial_crossings = find_crossing_pairs(left_points_R, left_edges) if left_edges else []
        right_initial_crossings = find_crossing_pairs(right_points_R, right_edges) if right_edges else []
        is_catastrophic = (lost_ratio > 0.5 or
                           len(left_initial_crossings) > 20 or
                           len(right_initial_crossings) > 20)

        if is_catastrophic:
            self.FRAME_DATA['current_left_crossings'] = left_initial_crossings
            self.FRAME_DATA['current_right_crossings'] = right_initial_crossings
        else:
            if left_edges:
                untangle(left_points_R, left_lost_mask, left_edges, matched_pairs, 'left')
            if right_edges:
                untangle(right_points_R, right_lost_mask, right_edges, matched_pairs, 'right')
            self.FRAME_DATA['current_left_crossings'] = (
                find_crossing_pairs(left_points_R, left_edges) if left_edges else [])
            self.FRAME_DATA['current_right_crossings'] = (
                find_crossing_pairs(right_points_R, right_edges) if right_edges else [])

        if not is_catastrophic:
            left_points_R = left_points_R.astype(np.float32)
            right_points_R = right_points_R.astype(np.float32)
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            l_pts_ud = cv2.undistortPoints(left_points_R, K1, D1, R=R1, P=P1).squeeze()
            r_pts_ud = cv2.undistortPoints(right_points_R, K2, D2, R=R2, P=P2).squeeze()
            points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

            base_3d = self.FRAME_DATA['base_3d_points']
            is_abnormal = False
            abnormal_reason = []
            if base_3d is not None and len(points_3d) == len(base_3d):
                z_diff = points_3d[:, 2] - base_3d[:, 2]
                if np.any(z_diff > 1.0):
                    is_abnormal = True
                    abnormal_reason.append("Z轴深度异常")
                if not is_abnormal and len(points_3d) >= 2:
                    distances = pdist(points_3d)
                    if np.any(distances < 0.3):
                        is_abnormal = True
                        abnormal_reason.append("点间距异常")
        else:
            is_abnormal = True
            abnormal_reason = ["灾难性熔断"]
            base_3d = self.FRAME_DATA['base_3d_points']

        # ── 连续异常帧计数与分级处理 ──
        if not is_abnormal:
            self.consecutive_abnormal_frames = 0
            self.FRAME_DATA['left_points_0_pre'] = left_points_R
            self.FRAME_DATA['right_points_0_pre'] = right_points_R
            return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask, False

        self.consecutive_abnormal_frames += 1

        if self.consecutive_abnormal_frames >= 5:
            self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
            self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
            self.consecutive_abnormal_frames = 0
            n_pts = len(base_3d) if base_3d is not None else len(self.FRAME_DATA['left_points_0_R'])
            return (self.FRAME_DATA['base_3d_points'].copy(),
                    self.FRAME_DATA['left_points_0_R'].copy(),
                    self.FRAME_DATA['right_points_0_R'].copy(),
                    np.ones(n_pts, dtype=bool),
                    np.ones(n_pts, dtype=bool),
                    True)
        else:
            prev_left = self.FRAME_DATA['left_points_0_pre']
            prev_right = self.FRAME_DATA['right_points_0_pre']
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1_prev, P2_prev, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            l_pts_ud_prev = cv2.undistortPoints(prev_left, K1, D1, R=R1, P=P1_prev).squeeze()
            r_pts_ud_prev = cv2.undistortPoints(prev_right, K2, D2, R=R2, P=P2_prev).squeeze()
            points_3d_prev = self.linear_triangulation(l_pts_ud_prev, r_pts_ud_prev, P1_prev, P2_prev)
            n_pts = len(base_3d) if base_3d is not None else len(prev_left)
            return (points_3d_prev,
                    prev_left.copy(),
                    prev_right.copy(),
                    np.ones(n_pts, dtype=bool),
                    np.ones(n_pts, dtype=bool),
                    True)

    # ──────────────────── 工具函数 ────────────────────

    def apply_roi_mask(self, points, mask):
        if mask is None:
            return points
        valid_points = []
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

    @staticmethod
    def extract_edges(points_2d):
        """从2D点集提取Delaunay三角剖分的去重边列表，并过滤幽灵长边。"""
        if len(points_2d) < 3:
            return []
        pts = np.asarray(points_2d, dtype=np.float64)
        try:
            tri = Delaunay(pts)
        except Exception:
            return []
        edges = set()
        for simplex in tri.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(a, c), max(a, c)))
        edge_list = list(edges)
        lengths = np.array([np.linalg.norm(pts[a] - pts[b]) for a, b in edge_list])
        threshold = np.median(lengths) * 1.3
        return [e for e, l in zip(edge_list, lengths) if l <= threshold]

    @staticmethod
    def segments_intersect(p1, p2, p3, p4):
        """判断线段 p1p2 与线段 p3p4 是否真正相交。"""
        if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or
                max(p3[0], p4[0]) < min(p1[0], p2[0]) or
                max(p1[1], p2[1]) < min(p3[1], p4[1]) or
                max(p3[1], p4[1]) < min(p1[1], p2[1])):
            return False
        eps = 3.0

        def cross2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross2d(p3, p4, p1)
        d2 = cross2d(p3, p4, p2)
        d3 = cross2d(p1, p2, p3)
        d4 = cross2d(p1, p2, p4)

        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
           ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
            return True
        return False

    def auto_match_points(self, left_points, right_points, left_pre, right_pre, max_dist=100):
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        left_ref = np.array(left_pre)
        right_ref = np.array(right_pre)
        num_markers = len(left_ref)

        def get_global_hungarian_match(ref_pts, det_pts, threshold):
            if len(ref_pts) == 0 or len(det_pts) == 0:
                return {}
            dist_matrix = cdist(ref_pts, det_pts)
            large_value = threshold * 10
            cost_matrix = dist_matrix.copy()
            cost_matrix[cost_matrix > threshold] = large_value
            n_ref = len(ref_pts)
            n_det = len(det_pts)
            if n_ref > n_det:
                padding = np.full((n_ref, n_ref - n_det), large_value)
                cost_matrix = np.hstack([cost_matrix, padding])
            elif n_det > n_ref:
                padding = np.full((n_det - n_ref, n_det), large_value)
                cost_matrix = np.vstack([cost_matrix, padding])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            final_map = {}
            for r_idx, d_idx in zip(row_ind, col_ind):
                if r_idx < n_ref and d_idx < n_det:
                    if dist_matrix[r_idx, d_idx] <= threshold:
                        final_map[r_idx] = d_idx
            return final_map

        l_map = get_global_hungarian_match(left_ref, left_points, max_dist)
        r_map = get_global_hungarian_match(right_ref, right_points, max_dist)
        matched = []
        for i in range(num_markers):
            l_idx = l_map.get(i, 100)
            r_idx = r_map.get(i, 100)
            matched.append((l_idx, r_idx))
        return matched

    def build_coordinate_system_pca(self, points_3d):
        centroid = np.mean(points_3d, axis=0)
        centered = points_3d - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        x_axis = eigenvectors[:, 0]
        y_axis = eigenvectors[:, 1]
        z_axis = eigenvectors[:, 2]
        if z_axis[2] < 0:
            z_axis = -z_axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        # 固定 X 轴大致朝向，防止特征向量符号不确定性导致 X-Y 平面 180° 翻转
        if x_axis[0] < 0:
            x_axis = -x_axis
            y_axis = -y_axis
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        return centroid, rotation_matrix

    def transform_to_local_coordinates(self, points):
        origin = self.FRAME_DATA['transform_origin']
        rotation = self.FRAME_DATA['transform_rotation']
        translated = points - origin
        return np.dot(translated, rotation)

    # ──────────────────── 显示 ────────────────────

    def display_frame(self, frame, left_pts=None, right_pts=None, left_lost=None, right_lost=None):
        display_img = frame.copy()

        # 绘制Delaunay拓扑网格连线
        left_edges = self.FRAME_DATA.get('left_edges') or []
        right_edges = self.FRAME_DATA.get('right_edges') or []
        if left_pts is not None and left_edges:
            crossing_left = set()
            for ei, ej in self.FRAME_DATA.get('current_left_crossings', []):
                crossing_left.add(ei)
                crossing_left.add(ej)
            for k, (a, b) in enumerate(left_edges):
                pa = (int(left_pts[a][0]), int(left_pts[a][1]))
                pb = (int(left_pts[b][0]), int(left_pts[b][1]))
                color = (0, 0, 220) if k in crossing_left else (200, 180, 80)
                cv2.line(display_img, pa, pb, color, 1, cv2.LINE_AA)
        if right_pts is not None and right_edges:
            crossing_right = set()
            for ei, ej in self.FRAME_DATA.get('current_right_crossings', []):
                crossing_right.add(ei)
                crossing_right.add(ej)
            for k, (a, b) in enumerate(right_edges):
                pa = (int(right_pts[a][0]), int(right_pts[a][1]))
                pb = (int(right_pts[b][0]), int(right_pts[b][1]))
                color = (0, 0, 220) if k in crossing_right else (80, 200, 80)
                cv2.line(display_img, pa, pb, color, 1, cv2.LINE_AA)

        # 绘制标记点
        if left_pts is not None:
            for i, pt in enumerate(left_pts):
                x, y = int(pt[0]), int(pt[1])
                color = (0, 165, 255) if (left_lost is not None and left_lost[i]) else (255, 0, 0)
                cv2.circle(display_img, (x, y), 4, color, -1)
        if right_pts is not None:
            for i, pt in enumerate(right_pts):
                x, y = int(pt[0]), int(pt[1])
                color = (0, 255, 255) if (right_lost is not None and right_lost[i]) else (0, 255, 0)
                cv2.circle(display_img, (x, y), 4, color, -1)

        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label_size = self.lbl_video.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)

    def update_3d_view(self, points_3d):
        if self.is_dragging:
            return
        if time.time() - self.drag_release_time < self.drag_cooldown:
            return
        # 正在更新基准帧时跳过，避免竞争条件
        if getattr(self, '_updating_base', False):
            return

        # 性能优化：只清除数据，不重建整个图
        if not hasattr(self, '_scatter_plot'):
            # 首次创建
            self.fig_3d.clear()
            ax = self.fig_3d.add_subplot(111, projection='3d')
            self.ax_3d = ax
            self._scatter_plot = None
            self._colorbar = None
        else:
            ax = self.ax_3d

        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)
            points[:, 1] = -points[:, 1]

            base_local = self.transform_to_local_coordinates(self.FRAME_DATA['base_3d_points'])
            base_local[:, 1] = -base_local[:, 1]

            displacement_vectors = points - base_local
            deformation = np.linalg.norm(displacement_vectors, axis=1)

            # 统计
            self.avg_displacement_vector = np.mean(displacement_vectors, axis=0)
            self.avg_displacement_magnitude = np.mean(deformation)
            self.max_displacement = np.max(deformation)
            self.min_displacement = np.min(deformation)

            # 更新散点图而不是重建
            if self._scatter_plot is None:
                self.fig_3d.clear()
                ax = self.fig_3d.add_subplot(111, projection='3d')
                self.ax_3d = ax
                self._scatter_plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                c=deformation, cmap='jet', s=50, vmin=0, vmax=1.5)
                self._colorbar = self.fig_3d.colorbar(self._scatter_plot, ax=ax, shrink=0.8)
                self._colorbar.set_label('Deformation (mm)', rotation=270, labelpad=15)
            else:
                self._scatter_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                self._scatter_plot.set_array(deformation)

            # 清除旧箭头
            if hasattr(self, '_force_arrow_plot') and self._force_arrow_plot is not None:
                try:
                    self._force_arrow_plot.remove()
                except (ValueError, AttributeError):
                    pass
                self._force_arrow_plot = None

            # 绘制预测力箭头
            if self.latest_predicted_force is not None:
                pred_force = self.latest_predicted_force[:3]
                force_mag = np.linalg.norm(pred_force)
                if force_mag > 0.5:
                    center = np.mean(points, axis=0)
                    force_dir = pred_force / force_mag
                    arrow_len = force_mag * 2.0
                    self._force_arrow_plot = ax.quiver(center[0], center[1], center[2],
                                     force_dir[0] * arrow_len, force_dir[1] * arrow_len, force_dir[2] * arrow_len,
                                     color='blue', arrow_length_ratio=0.2, linewidth=3,
                                     label=f'预测力 ({force_mag:.2f}N)')
                    ax.legend(loc='upper left', fontsize=8)
        else:
            points = points_3d.copy()
            points[:, 0] = -points[:, 0]
            points[:, 1] = -points[:, 1]
            if self._scatter_plot is None:
                self.fig_3d.clear()
                ax = self.fig_3d.add_subplot(111, projection='3d')
                self.ax_3d = ax
                self._scatter_plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)
            else:
                self._scatter_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Frame {self.current_frame_idx}')

        if len(points) > 0:
            ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])

        if self.view_saved:
            try:
                ax.view_init(elev=self.view_elev, azim=self.view_azim, roll=self.view_roll)
            except TypeError:
                ax.view_init(elev=self.view_elev, azim=self.view_azim)
        else:
            try:
                ax.view_init(elev=90, azim=0, roll=-90)
            except TypeError:
                ax.view_init(elev=90, azim=0)

        # 使用blit加速渲染
        self.canvas_3d.draw_idle()

    # ──────────────────── 播放控制 ────────────────────

    def toggle_play_pause(self):
        if self.h5_mode:
            # HDF5回放模式
            self.h5_playing = not self.h5_playing
            if self.h5_playing:
                self.btn_play_pause.setText("暂停")
                QTimer.singleShot(33, self.h5_next_frame)
            else:
                self.btn_play_pause.setText("播放")
        else:
            # 实时采集模式
            if self.is_playing:
                self.is_playing = False
                self.btn_play_pause.setText("播放")
                self.status_bar.showMessage("已暂停")
            else:
                self.is_playing = True
                self.btn_play_pause.setText("暂停")
                self.status_bar.showMessage("播放中...")

    def h5_next_frame(self):
        """HDF5回放下一帧"""
        if not self.h5_playing or not self.h5_mode:
            return

        n_frames = len(self.h5_vision_data['timestamp'])
        if self.h5_current_idx < n_frames - 1:
            self.h5_current_idx += 1
            self.update_h5_display()
            QTimer.singleShot(33, self.h5_next_frame)
        else:
            self.h5_playing = False
            self.btn_play_pause.setText("播放")

    def update_playback_speed(self, fps):
        if self.camera_thread is not None:
            self.camera_thread.target_fps = fps

    def reset_first_frame(self):
        self.current_frame_idx = 0
        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
        self.consecutive_abnormal_frames = 0

        # 重置scatter对象
        self._scatter_plot = None
        self._colorbar = None
        self._force_arrow_plot = None

        self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        self.status_bar.showMessage("已重置参考点")

    def set_current_as_base(self):
        if not self.FRAME_DATA['initialized']:
            self.status_bar.showMessage("请先初始化首帧")
            return
        left_pts = self.FRAME_DATA.get('left_points_0_pre')
        right_pts = self.FRAME_DATA.get('right_points_0_pre')
        if left_pts is None or right_pts is None:
            self.status_bar.showMessage("当前帧数据不可用")
            return

        # 暂停3D更新，避免竞争条件
        self._updating_base = True

        self.FRAME_DATA['left_points_0_R'] = left_pts.copy()
        self.FRAME_DATA['right_points_0_R'] = right_pts.copy()

        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']
        w = self.FRAME_DATA['mirror_axis'] * 2
        h = self.FRAME_DATA.get('frame_height', 480)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        l_pts_ud = cv2.undistortPoints(left_pts, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_pts, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)
        self.FRAME_DATA['base_3d_points'] = points_3d

        # 更新基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(left_pts.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(right_pts.astype(np.float64))

        # 重新计算坐标变换矩阵（与新基准帧匹配）
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix

        # 重置scatter对象，因为基准帧改变了
        self._scatter_plot = None
        self._colorbar = None
        if hasattr(self, '_force_arrow_plot') and self._force_arrow_plot is not None:
            try:
                self._force_arrow_plot.remove()
            except (ValueError, AttributeError):
                pass
            self._force_arrow_plot = None
        self.fig_3d.clear()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d.draw_idle()

        # 自动调零力传感器
        self.auto_zero_ft()

        self._updating_base = False
        self.consecutive_abnormal_frames = 0

        self.status_bar.showMessage(f"已将当前帧设置为新的基准帧并调零力传感器（帧 {self.current_frame_idx}）")

    # ──────────────────── 3D 视图鼠标事件 ────────────────────

    def on_3d_mouse_press(self, event):
        self.is_dragging = True
        self._was_playing_before_drag = self.is_playing
        if self.is_playing:
            self.is_playing = False

    def on_3d_mouse_motion(self, event):
        if self.is_dragging and hasattr(self, 'ax_3d'):
            self.view_elev = self.ax_3d.elev
            self.view_azim = self.ax_3d.azim
            self.view_roll = getattr(self.ax_3d, 'roll', 0)
            self.view_saved = True

    def on_3d_mouse_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            self.drag_release_time = time.time()
            if hasattr(self, 'ax_3d'):
                self.view_elev = self.ax_3d.elev
                self.view_azim = self.ax_3d.azim
                self.view_roll = getattr(self.ax_3d, 'roll', 0)
                self.view_saved = True
            if self._was_playing_before_drag:
                self.is_playing = True
                self._was_playing_before_drag = False

    # ──────────────────── 关闭 ────────────────────

    # ──────────────────── HDF5回放功能 ────────────────────

    def load_h5_file(self):
        """加载HDF5文件进行回放"""
        file_path, _ = QFileDialog.getOpenFileName(self, '选择HDF5文件', '', 'HDF5 Files (*.h5 *.hdf5)')
        if not file_path:
            return

        try:
            if self.h5_file:
                self.h5_file.close()

            self.h5_file = h5py.File(file_path, 'r')

            # 读取数据
            self.h5_vision_data = {
                'timestamp': self.h5_file['/vision/timestamp'][:],
                'xyz': self.h5_file['/vision/xyz'][:],
                'predicted_force': self.h5_file['/vision/predicted_force'][:]
            }
            self.h5_force_data = {
                'timestamp': self.h5_file['/force/timestamp'][:],
                'values': self.h5_file['/force/values'][:]
            }
            self.h5_xyz_ref = self.h5_file['/reference/xyz_ref'][:]

            # 同步力数据
            self.sync_h5_force_to_vision()

            # 切换到HDF5模式
            self.h5_mode = True
            self.h5_current_idx = 0
            # 重置绘图缓存
            if hasattr(self, '_h5_scatter_ref'):
                del self._h5_scatter_ref
            if hasattr(self, '_h5_force_lines_actual'):
                del self._h5_force_lines_actual
            # 重置实时力曲线的 blit 状态
            if hasattr(self, '_force_lines_actual'):
                del self._force_lines_actual
            self._force_plot_ready = False
            self.ax_3d.clear()
            self.btn_play_pause.setEnabled(True)
            self.btn_play_pause.setText('播放')
            self.status_bar.showMessage(f'已加载HDF5: {os.path.basename(file_path)}')

            # 显示第一帧
            self.update_h5_display()

        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载HDF5失败：{str(e)}')

    def sync_h5_force_to_vision(self):
        """同步力数据到视觉帧"""
        vision_ts = self.h5_vision_data['timestamp']
        force_ts = self.h5_force_data['timestamp']
        force_vals = self.h5_force_data['values']

        synced_force = np.zeros((len(vision_ts), 6))
        for i, vts in enumerate(vision_ts):
            idx = np.argmin(np.abs(force_ts - vts))
            synced_force[i] = force_vals[idx]

        self.h5_vision_data['synced_force'] = synced_force

    def update_h5_display(self):
        """更新HDF5回放显示"""
        if not self.h5_mode or self.h5_vision_data is None:
            return

        idx = self.h5_current_idx
        n_frames = len(self.h5_vision_data['timestamp'])
        self.lbl_frame_info.setText(f'帧: {idx+1}/{n_frames}')

        # 更新点云
        self.update_h5_point_cloud(idx)

        # 力数据
        actual = self.h5_vision_data['synced_force'][idx]
        pred = self.h5_vision_data['predicted_force'][idx]

        # 更新力曲线子图（HDF5模式：显示完整曲线 + 移动竖线）
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        if not hasattr(self, '_h5_force_lines_actual'):
            self._h5_force_lines_actual = []
            self._h5_force_lines_pred = []
            self._h5_force_vlines = []
            x = np.arange(n_frames)
            for i, ax in enumerate(self.axes_force.flat):
                ax.clear()
                line_actual, = ax.plot(x, self.h5_vision_data['synced_force'][:, i], 'r-', label='实际', linewidth=1.5, alpha=0.7)
                line_pred, = ax.plot(x, self.h5_vision_data['predicted_force'][:, i], 'b-', label='预测', linewidth=1.5, alpha=0.7)
                vline = ax.axvline(idx, color='green', linestyle='--', linewidth=2, alpha=0.8)
                ax.set_title(f'{labels[i]}: 实际={actual[i]:.2f}, 预测={pred[i]:.2f}', fontsize=10)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, n_frames])
                self._h5_force_lines_actual.append(line_actual)
                self._h5_force_lines_pred.append(line_pred)
                self._h5_force_vlines.append(vline)
            self.fig_force.tight_layout()
        else:
            for i, ax in enumerate(self.axes_force.flat):
                self._h5_force_vlines[i].set_xdata([idx, idx])
                ax.set_title(f'{labels[i]}: 实际={actual[i]:.2f}, 预测={pred[i]:.2f}', fontsize=10)

        self.canvas_force.draw_idle()

    def update_h5_point_cloud(self, idx):
        """更新HDF5点云显示（带力箭头）"""
        xyz = self.h5_vision_data['xyz'][idx].copy()
        # 绕Z轴旋转180°
        xyz[:, 0] = -xyz[:, 0]
        xyz[:, 1] = -xyz[:, 1]

        predicted_force = self.h5_vision_data['predicted_force'][idx]

        # 首次创建或需要重建
        if not hasattr(self, '_h5_scatter_ref'):
            self.ax_3d.clear()
            self._h5_scatter_ref = None
            self._h5_scatter_cur = None
            self._h5_force_quiver = None

            if self.h5_xyz_ref is not None and len(self.h5_xyz_ref) > 0:
                self._h5_scatter_ref = self.ax_3d.scatter(
                    self.h5_xyz_ref[:, 0], self.h5_xyz_ref[:, 1], self.h5_xyz_ref[:, 2],
                    c='gray', s=1, alpha=0.3, label='参考')

            if len(xyz) > 0:
                self._h5_scatter_cur = self.ax_3d.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c='blue', s=10, alpha=0.8, label='当前')

            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('Z')
            self.ax_3d.legend()
        else:
            # 更新当前点云数据
            if len(xyz) > 0:
                if self._h5_scatter_cur is not None:
                    self._h5_scatter_cur._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
                else:
                    self._h5_scatter_cur = self.ax_3d.scatter(
                        xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        c='blue', s=10, alpha=0.8, label='当前')

            # 移除旧力箭头
            if self._h5_force_quiver is not None:
                self._h5_force_quiver.remove()
                self._h5_force_quiver = None

        # 绘制力箭头
        if len(xyz) > 0:
            center = np.mean(xyz, axis=0)
            force_vec = predicted_force[:3]
            force_mag = np.linalg.norm(force_vec)

            if force_mag > 0.5:
                force_dir = force_vec / force_mag
                arrow_len = force_mag * 2.0
                self._h5_force_quiver = self.ax_3d.quiver(
                    center[0], center[1], center[2],
                    force_dir[0] * arrow_len, force_dir[1] * arrow_len, force_dir[2] * arrow_len,
                    color='red', arrow_length_ratio=0.2, linewidth=3,
                    label=f'预测力 ({force_mag:.2f}N)')

        self.ax_3d.set_title(f'点云 + 预测力 (帧 {idx+1})')

        # 设置坐标轴范围
        if self.h5_xyz_ref is not None and len(self.h5_xyz_ref) > 0:
            all_points = np.vstack([self.h5_xyz_ref, xyz]) if len(xyz) > 0 else self.h5_xyz_ref
        elif len(xyz) > 0:
            all_points = xyz
        else:
            all_points = np.array([[0, 0, 0]])

        margin = 5
        self.ax_3d.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
        self.ax_3d.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
        self.ax_3d.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])

        self.canvas_3d.draw_idle()

    def closeEvent(self, event):
        if self.h5_file:
            self.h5_file.close()
        if self.is_recording:
            self.stop_recording()
        self.force_plot_timer.stop()
        if self.ft_node is not None:
            self.ft_subscription = None
            try:
                self.ft_node.Shutdown()
            except Exception:
                pass
            self.ft_node = None
        if self.process_thread is not None:
            self.process_thread.running = False
            self.process_thread.has_new_frame.set()
        if self.camera_thread is not None:
            self.camera_thread.running = False
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        event.accept()


# ─────────────────────── main ───────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    window = V7MainWindow()
    window.show()
    ret = app.exec_()
    os._exit(ret)
