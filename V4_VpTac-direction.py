# -*- coding: utf-8 -*-
'''
视频/摄像头标记点跟踪与三维重建播放器 + 压电触觉传感器数据采集
功能说明：
1. 支持读取视频文件或USB摄像头实时采集
2. 逐帧进行标记点检测和三维重建
3. 启动前弹出文件选择窗口：标定文件夹、参数JSON文件、首帧ROI和匹配数据保存文件夹
4. GUI中同时显示视频帧、三维点云和触觉传感器信号
5. 支持暂停/播放，支持手动调整三维点云观察角度
6. 视觉和触觉信号时间同步
'''
import sys
import os
import json
import time
import threading
import numpy as np
import cv2
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QGroupBox, QStatusBar, QSlider, QSpinBox, QSplitter,
                             QComboBox, QCheckBox, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import Rbf
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pyqtgraph as pg


class SerialReceiver(QThread):
    """串口数据接收线程"""
    data_received = pyqtSignal(bytes, float)  # 数据和时间戳

    def __init__(self, port='COM32', baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.buffer = b''
        self.start_time = None 

    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.start_time = time.time()
            while self.running:
                if self.serial.in_waiting > 0:
                    self.buffer += self.serial.read(self.serial.in_waiting)
                    self.process_data()
        except serial.SerialException as e:
            print(f"串口错误: {e}")

    def stop(self):
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.quit()
        self.wait()

    def process_data(self):
            FRAME_LENGTH = 29
            while True:
                aaaa_index = self.buffer.find(b'\xaa\xaa')
                if aaaa_index == -1 or len(self.buffer) < aaaa_index + FRAME_LENGTH:
                    break # 长度不够，正常等待

                if aaaa_index + FRAME_LENGTH <= len(self.buffer):
                    if self.buffer[aaaa_index + FRAME_LENGTH - 2: aaaa_index + FRAME_LENGTH] == b'\xff\xff':
                        group_data = self.buffer[aaaa_index + 2: aaaa_index + FRAME_LENGTH - 2]
                        self.data_received.emit(group_data)
                        self.buffer = self.buffer[aaaa_index + FRAME_LENGTH:]
                    else:
                        # 🚨 关键修复：数据错位了，把这个作废的包头扔掉，缓冲区往前推！
                        self.buffer = self.buffer[aaaa_index + 2:]
                else:
                    break


class CameraThread(QThread):
    """摄像头采集线程"""
    frame_signal = pyqtSignal(np.ndarray)

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
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.frame_signal.emit(frame)

            elapsed = time.time() - start_time
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        if self.video_cap:
            self.video_cap.release()

    def stop(self):
        self.running = False
        self.wait()


class FrameProcessThread(QThread):
    """帧处理线程 - 将耗时的检测和重建放到后台"""
    result_signal = pyqtSignal(object, object, object, object, object, object)  # frame, points_3d, left_pts, right_pts, left_lost, right_lost

    def __init__(self):
        super().__init__()
        self.running = True
        self.frame_queue = []
        self.lock = threading.Lock()
        self.process_func = None
        self.has_new_frame = threading.Event()

    def set_processor(self, process_func):
        self.process_func = process_func

    def add_frame(self, frame):
        with self.lock:
            # 只保留最新帧，丢弃旧帧
            self.frame_queue = [frame]
        self.has_new_frame.set()

    def run(self):
        while self.running:
            self.has_new_frame.wait(timeout=0.1)
            if not self.running:
                break

            frame = None
            with self.lock:
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
            self.has_new_frame.clear()

            if frame is not None and self.process_func is not None:
                try:
                    points_3d, left_pts, right_pts, left_lost, right_lost = self.process_func(frame)
                    self.result_signal.emit(frame, points_3d, left_pts, right_pts, left_lost, right_lost)
                except Exception as e:
                    pass

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        self.wait()


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


class VideoPointCloudPlayer(QMainWindow):
    """视频/摄像头点云播放器主窗口 + 触觉传感器"""

    # 配置文件路径（保存上次选择的输入源配置）
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_input_config.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉触觉融合系统 - 标记点跟踪与触觉信号采集")
        self.setGeometry(50, 50, 1600, 950)

        # 输入模式: 'video' 或 'camera'
        self.input_mode = None

        # 路径变量
        self.video_path = None
        self.calib_dir = None
        self.params_json_path = None
        self.data_save_dir = None

        # 视频相关
        self.video_cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 60

        # 摄像头相关
        self.camera_thread = None
        self.camera_index = 0
        self.rotate_180 = True
        self.latest_camera_frame = None

        # 帧处理线程（摄像头模式）
        self.process_thread = None
        self.last_3d_update_time = 0
        self.min_3d_update_interval = 0.1  # 3D视图最小更新间隔（秒）

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

        # 3D视角
        self.view_elev = 90
        self.view_azim = 0
        self.view_roll = -90  # 添加roll参数，初始在XY平面顺时针旋转90°
        self.view_saved = False
        self.is_dragging = False
        self.drag_release_time = 0  # 拖拽释放时间
        self.drag_cooldown = 0.5  # 拖拽后冷却时间（秒）- 增加到0.5秒
        self._was_playing_before_drag = False  # 记录拖拽前是否在播放

        # 播放控制
        self.is_playing = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.on_timer_tick)

        # 触觉传感器相关
        self.serial_receiver = None
        self.tactile_start_time = None  # 触觉数据采集开始时间
        self.visual_start_time = None   # 视觉数据采集开始时间
        self.data_ptac_adc1 = []
        self.data_ptac_adc2 = []
        self.data_ptac_adc3 = []
        self.data_ptac_adc4 = []
        self.data_ptac_adc5 = []
        self.tactile_timestamps = []  # 触觉数据时间戳
        self.visual_timestamps = []   # 视觉数据时间戳

        # 触觉信号更新定时器
        self.tactile_timer = QTimer(self)
        self.tactile_timer.timeout.connect(self.update_tactile_plot)
        self.tactile_timer.start(50)

        # 位移统计数据
        self.avg_displacement_vector = np.array([0.0, 0.0, 0.0])  # 平均位移向量
        self.avg_displacement_magnitude = 0.0  # 平均位移大小
        self.max_displacement = 0.0  # 最大位移
        self.min_displacement = 0.0  # 最小位移
        self.current_points_3d = None  # 当前帧3D点
        self.consecutive_abnormal_frames = 0  # 连续异常帧计数器

        # 初始化界面
        self.init_ui()

        # 启动时尝试自动加载上次的输入源配置
        QTimer.singleShot(100, self.auto_load_last_config)

    def init_ui(self):
        """初始化界面"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        self.btn_select_input = QPushButton("选择输入源")
        self.btn_select_input.clicked.connect(self.select_input_mode)

        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)

        self.btn_reset = QPushButton("重置首帧")
        self.btn_reset.clicked.connect(self.reset_first_frame)
        self.btn_reset.setEnabled(False)

        self.btn_set_as_base = QPushButton("设置为首帧")
        self.btn_set_as_base.clicked.connect(self.set_current_as_base)
        self.btn_set_as_base.setEnabled(False)

        self.lbl_frame_info = QLabel("帧: 0 / 0")

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 120)
        self.spin_speed.setValue(60)
        self.spin_speed.setPrefix("速度: ")
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.valueChanged.connect(self.update_playback_speed)

        # 最大匹配距离控件
        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 500)
        self.spin_max_dist.setValue(50)
        self.spin_max_dist.setPrefix("匹配距离: ")
        self.spin_max_dist.valueChanged.connect(self.update_max_dist)

        # 摄像头控件
        self.lbl_camera = QLabel("摄像头:")
        self.spin_camera_idx = QSpinBox()
        self.spin_camera_idx.setRange(0, 10)
        self.spin_camera_idx.setValue(0)
        self.spin_camera_idx.setPrefix("ID: ")

        self.chk_rotate = QCheckBox("旋转180°")
        self.chk_rotate.setChecked(True)

        control_layout.addWidget(self.btn_select_input)
        control_layout.addWidget(self.btn_play_pause)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.btn_set_as_base)
        control_layout.addWidget(self.lbl_frame_info)
        control_layout.addWidget(self.spin_speed)
        control_layout.addWidget(self.spin_max_dist)
        control_layout.addWidget(self.lbl_camera)
        control_layout.addWidget(self.spin_camera_idx)
        control_layout.addWidget(self.chk_rotate)
        control_layout.addStretch()

        # 串口控制区域
        self.lbl_serial = QLabel("串口:")
        self.cbb_serial_port = QComboBox()
        self.cbb_serial_port.addItems(self.get_serial_ports())
        self.cbb_serial_baud = QComboBox()
        self.cbb_serial_baud.addItems(["921600", "115200"])
        self.cbb_adc_group = QComboBox()
        self.cbb_adc_group.addItems(["ADC1", "ADC2", "ADC3", "ADC4", "ADC5"])
        self.btn_serial_connect = QPushButton("连接串口")
        self.btn_serial_connect.clicked.connect(self.start_serial_reading)
        self.btn_serial_disconnect = QPushButton("断开串口")
        self.btn_serial_disconnect.clicked.connect(self.stop_serial_reading)
        self.btn_refresh_ports = QPushButton("刷新")
        self.btn_refresh_ports.clicked.connect(self.refresh_serial_ports)

        control_layout.addWidget(self.lbl_serial)
        control_layout.addWidget(self.cbb_serial_port)
        control_layout.addWidget(self.cbb_serial_baud)
        control_layout.addWidget(self.cbb_adc_group)
        control_layout.addWidget(self.btn_serial_connect)
        control_layout.addWidget(self.btn_serial_disconnect)
        control_layout.addWidget(self.btn_refresh_ports)

        # 通道选择控件（3个通道）
        self.lbl_channels = QLabel("显示通道:")
        self.spin_ch1 = QSpinBox()
        self.spin_ch1.setRange(1, 8)
        self.spin_ch1.setValue(1)
        self.spin_ch1.setPrefix("CH")
        self.spin_ch2 = QSpinBox()
        self.spin_ch2.setRange(1, 8)
        self.spin_ch2.setValue(2)
        self.spin_ch2.setPrefix("CH")
        self.spin_ch3 = QSpinBox()
        self.spin_ch3.setRange(1, 8)
        self.spin_ch3.setValue(3)
        self.spin_ch3.setPrefix("CH")

        control_layout.addWidget(self.lbl_channels)
        control_layout.addWidget(self.spin_ch1)
        control_layout.addWidget(self.spin_ch2)
        control_layout.addWidget(self.spin_ch3)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 进度条（仅视频模式使用）
        slider_group = QGroupBox("进度")
        slider_layout = QHBoxLayout()
        self.slider_progress = QSlider(Qt.Horizontal)
        self.slider_progress.setRange(0, 100)
        self.slider_progress.setValue(0)
        self.slider_progress.sliderPressed.connect(self.on_slider_pressed)
        self.slider_progress.sliderReleased.connect(self.on_slider_released)
        self.slider_progress.valueChanged.connect(self.on_slider_changed)
        self.slider_dragging = False
        slider_layout.addWidget(self.slider_progress)
        slider_group.setLayout(slider_layout)
        main_layout.addWidget(slider_group)

        # 主显示区域 - 使用垂直QSplitter分为上下两部分
        main_splitter = QSplitter(Qt.Vertical)

        # 上部：视觉显示区域（视频+3D点云）
        visual_splitter = QSplitter(Qt.Horizontal)

        # 左侧：2D视频显示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.lbl_video = QLabel("视频显示区域")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(640, 400)
        self.lbl_video.setStyleSheet("background-color: #2a2a2a; color: white;")
        left_layout.addWidget(self.lbl_video)
        visual_splitter.addWidget(left_widget)

        # 右侧：3D点云显示
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

        # 下部：触觉信号显示区域
        tactile_widget = QWidget()
        tactile_layout = QVBoxLayout(tactile_widget)
        tactile_layout.setContentsMargins(5, 5, 5, 5)

        # 使用pyqtgraph绘制触觉信号（3个通道，横向排列）
        self.tactile_plot_widget = pg.GraphicsLayoutWidget()
        self.tactile_plot_widget.setBackground('w')
        self.tactile_plot_widget.setMinimumHeight(200)

        self.tactile_plots = []
        self.tactile_curves = []
        for i in range(3):
            plot_item = self.tactile_plot_widget.addPlot(row=0, col=i, title=f"CH:{i + 1}")
            plot_item.setYRange(-3, 3)  # 纵坐标范围 -3 到 3
            plot_item.setTitle(f"CH:{i + 1}", color='k', size='10pt')
            plot_item.getAxis('left').setPen(pg.mkPen(color='k'))
            plot_item.getAxis('bottom').setPen(pg.mkPen(color='k'))
            plot_item.getAxis('left').setTextPen(pg.mkPen(color='k'))
            plot_item.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
            self.tactile_plots.append(plot_item)
            curve = plot_item.plot(pen='b')
            self.tactile_curves.append(curve)

        tactile_layout.addWidget(self.tactile_plot_widget)
        main_splitter.addWidget(tactile_widget)

        # 设置上下分割比例
        main_splitter.setSizes([550, 250])
        main_layout.addWidget(main_splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择视频和配置文件")

        # 3D视图鼠标事件
        self.canvas_3d.mpl_connect('button_press_event', self.on_3d_mouse_press)
        self.canvas_3d.mpl_connect('button_release_event', self.on_3d_mouse_release)
        self.canvas_3d.mpl_connect('motion_notify_event', self.on_3d_mouse_motion)

    def save_input_config(self):
        """保存当前输入源配置到文件"""
        config = {
            'input_mode': self.input_mode,
            'video_path': self.video_path,
            'calib_dir': self.calib_dir,
            'params_json_path': self.params_json_path,
            'data_save_dir': self.data_save_dir,
            'camera_index': self.spin_camera_idx.value(),
            'rotate_180': self.chk_rotate.isChecked()
        }
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"配置已保存: {self.CONFIG_FILE}")
        except Exception as e:
            print(f"保存配置失败: {e}")

    def load_input_config(self):
        """从文件加载上次的输入源配置，返回是否成功加载"""
        if not os.path.exists(self.CONFIG_FILE):
            return False
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 验证配置有效性
            input_mode = config.get('input_mode')
            if input_mode not in ['video', 'camera']:
                return False

            # 验证必要路径存在
            calib_dir = config.get('calib_dir')
            data_save_dir = config.get('data_save_dir')
            if not calib_dir or not os.path.exists(calib_dir):
                return False
            if not data_save_dir or not os.path.exists(data_save_dir):
                return False

            # 视频模式需要验证视频文件
            if input_mode == 'video':
                video_path = config.get('video_path')
                if not video_path or not os.path.exists(video_path):
                    return False
                self.video_path = video_path

            # 加载配置
            self.input_mode = input_mode
            self.calib_dir = calib_dir
            self.data_save_dir = data_save_dir
            self.params_json_path = config.get('params_json_path')

            # 加载摄像头设置
            if 'camera_index' in config:
                self.spin_camera_idx.setValue(config['camera_index'])
            if 'rotate_180' in config:
                self.chk_rotate.setChecked(config['rotate_180'])

            print(f"已加载上次配置: {input_mode} 模式")
            return True
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False

    def auto_load_last_config(self):
        """自动加载上次配置并启动"""
        if not self.load_input_config():
            self.status_bar.showMessage("就绪 - 请选择输入源")
            return

        # 根据模式自动启动
        if self.input_mode == 'video':
            self.lbl_camera.setVisible(False)
            self.spin_camera_idx.setVisible(False)
            self.chk_rotate.setVisible(False)
            self.slider_progress.setEnabled(True)
            self.load_all_configs()
        elif self.input_mode == 'camera':
            self.lbl_camera.setVisible(True)
            self.spin_camera_idx.setVisible(True)
            self.chk_rotate.setVisible(True)
            self.slider_progress.setEnabled(False)
            self.load_configs_for_camera()

    def select_input_mode(self):
        """选择输入模式：视频文件或USB摄像头"""
        # 停止当前运行的处理线程
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None

        # 停止当前运行的摄像头线程
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

        # 停止播放
        self.pause_playback()

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("选择输入源")
        msg_box.setText("请选择输入源类型：")
        btn_video = msg_box.addButton("视频文件", QMessageBox.ActionRole)
        btn_camera = msg_box.addButton("USB摄像头", QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton("取消", QMessageBox.RejectRole)
        msg_box.exec_()

        clicked = msg_box.clickedButton()
        if clicked == btn_video:
            self.input_mode = 'video'
            self.select_files_for_video()
        elif clicked == btn_camera:
            self.input_mode = 'camera'
            self.select_files_for_camera()
        # 取消时不再自动关闭程序，允许用户单独测试压电传感器

    def select_files_for_video(self):
        """视频模式：选择视频文件和配置"""
        # 1. 选择视频文件
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if not video_path:
            QMessageBox.warning(self, "警告", "未选择视频文件")
            return
        self.video_path = video_path

        # 选择公共配置
        if not self.select_common_configs():
            return

        # 隐藏摄像头控件，显示进度条
        self.lbl_camera.setVisible(False)
        self.spin_camera_idx.setVisible(False)
        self.chk_rotate.setVisible(False)
        self.slider_progress.setEnabled(True)

        # 加载配置
        self.load_all_configs()

        # 保存当前配置供下次启动使用
        self.save_input_config()

    def select_files_for_camera(self):
        """摄像头模式：选择配置文件"""
        # 选择公共配置
        if not self.select_common_configs():
            return

        # 显示摄像头控件，禁用进度条
        self.lbl_camera.setVisible(True)
        self.spin_camera_idx.setVisible(True)
        self.chk_rotate.setVisible(True)
        self.slider_progress.setEnabled(False)

        # 加载配置（不加载视频）
        self.load_configs_for_camera()

        # 保存当前配置供下次启动使用
        self.save_input_config()

    def select_common_configs(self):
        """选择公共配置文件"""
        # 2. 选择标定文件夹
        calib_dir = QFileDialog.getExistingDirectory(
            self, "选择标定参数文件夹（包含K1.txt, K2.txt等）", "")
        if not calib_dir:
            QMessageBox.warning(self, "警告", "未选择标定文件夹")
            return False
        self.calib_dir = calib_dir

        # 3. 选择参数JSON文件
        params_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像处理参数文件 (marker_params.json)", "",
            "JSON文件 (*.json);;所有文件 (*.*)")
        if not params_path:
            QMessageBox.warning(self, "警告", "未选择参数文件，将使用默认参数")
            self.params_json_path = None
        else:
            self.params_json_path = params_path

        # 4. 选择保存ROI和匹配数据的文件夹
        data_dir = QFileDialog.getExistingDirectory(
            self, "选择data文件夹（包含roi_masks.npz，同级目录应有result文件夹）", "")
        if not data_dir:
            QMessageBox.warning(self, "警告", "未选择数据文件夹")
            return False
        self.data_save_dir = data_dir

        return True

    def load_configs_for_camera(self):
        """摄像头模式：加载配置"""
        # 加载标定参数
        if not self.load_calibration():
            return

        # 加载检测器
        self.load_detector()

        # 加载ROI和首帧参考数据
        if not self.load_reference_data():
            return

        # 启动摄像头
        self.start_camera()

    def load_reference_data(self):
        """加载ROI和首帧参考数据"""
        # 加载ROI
        roi_file = os.path.join(self.data_save_dir, "roi_masks.npz")
        if os.path.exists(roi_file):
            try:
                with np.load(roi_file) as data:
                    left_mask = data['left_mask']
                    right_mask = data['right_mask']
                    if left_mask.ndim == 3: left_mask = left_mask[:, :, 0]
                    if right_mask.ndim == 3: right_mask = right_mask[:, :, 0]
                    self.drawing['left']['mask'] = left_mask
                    self.drawing['right']['mask'] = right_mask
            except Exception as e:
                QMessageBox.warning(self, "警告", f"ROI加载失败: {e}")
                return False
        else:
            QMessageBox.warning(self, "警告", f"未找到ROI文件: {roi_file}")
            return False

        # 从matched_points.npz加载mirror_axis
        mirror_axis = 640  # 默认值
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
            except Exception as e:
                pass

        # 从result文件夹加载首帧数据
        result_dir = os.path.join(self.data_save_dir, "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")

        if not os.path.exists(first_frame_file):
            QMessageBox.warning(self, "警告", f"未找到首帧数据文件: {first_frame_file}\n请先用V4_test处理图片序列")
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

        # 标定参数
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']

        # 使用默认图像尺寸进行立体校正
        h, w = 480, 1280  # 默认尺寸，会在收到第一帧时更新
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

        pts1 = cv2.undistortPoints(pts1_R, K1, D1, R=R1, P=P1).squeeze()
        pts2 = cv2.undistortPoints(pts2_R, K2, D2, R=R2, P=P2).squeeze()

        # 更新全局数据
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
        # 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        # 构建坐标系
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix

        return True

    def start_camera(self):
        """启动摄像头"""
        self.camera_index = self.spin_camera_idx.value()
        self.rotate_180 = self.chk_rotate.isChecked()

        # 记录视觉开始时间（用于与触觉同步）
        self.visual_start_time = time.time()

        # 启动帧处理线程
        self.process_thread = FrameProcessThread()
        self.process_thread.set_processor(self.process_frame)
        self.process_thread.result_signal.connect(self.on_process_result)
        self.process_thread.start()

        # 启动摄像头线程
        self.camera_thread = CameraThread(
            camera_index=self.camera_index,
            fps=self.spin_speed.value(),
            rotate_180=self.rotate_180
        )
        self.camera_thread.frame_signal.connect(self.on_camera_frame)
        self.camera_thread.start()

        # 启用控制按钮
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("暂停")
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.is_playing = True

        self.status_bar.showMessage(f"摄像头已启动 (ID: {self.camera_index})")

    def on_camera_frame(self, frame):
        """处理摄像头帧 - 仅分发到处理线程，不直接显示"""
        if not self.FRAME_DATA['initialized']:
            return

        self.latest_camera_frame = frame
        self.current_frame_idx += 1

        # 更新mirror_axis（如果图像尺寸变化）
        h, w = frame.shape[:2]
        if self.FRAME_DATA['mirror_axis'] is None or self.FRAME_DATA['mirror_axis'] != w // 2:
            # 重新计算投影矩阵
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            self.FRAME_DATA['P1'] = P1
            self.FRAME_DATA['P2'] = P2

        # 每10帧更新一次帧计数显示，减少GUI更新频率
        if self.current_frame_idx % 10 == 0:
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx}")

        if self.is_playing and self.process_thread is not None:
            # 将帧发送到处理线程（异步处理）
            self.process_thread.add_frame(frame.copy())
        else:
            # 暂停时仅显示原始帧
            self.display_frame(frame)

    def on_process_result(self, frame, points_3d, left_pts, right_pts, left_lost, right_lost):
        """处理线程返回结果 - 更新显示"""
        if points_3d is not None:
            # 更新带标记点的帧显示
            self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)

            # 检查是否正在拖动或在冷却期内，如果是则跳过3D更新
            if self.is_dragging:
                return
            if time.time() - self.drag_release_time < self.drag_cooldown:
                return

            # 限制3D视图更新频率
            current_time = time.time()
            if current_time - self.last_3d_update_time >= self.min_3d_update_interval:
                self.update_3d_view(points_3d)
                self.last_3d_update_time = current_time

    def select_files_on_startup(self):
        """启动时选择必要的文件（视频模式）"""
        # 1. 选择视频文件
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if not video_path:
            QMessageBox.warning(self, "警告", "未选择视频文件，程序将退出")
            QTimer.singleShot(100, self.close)
            return
        self.video_path = video_path

        # 2. 选择标定文件夹
        calib_dir = QFileDialog.getExistingDirectory(
            self, "选择标定参数文件夹（包含K1.txt, K2.txt等）", "")
        if not calib_dir:
            QMessageBox.warning(self, "警告", "未选择标定文件夹，程序将退出")
            QTimer.singleShot(100, self.close)
            return
        self.calib_dir = calib_dir

        # 3. 选择参数JSON文件
        params_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像处理参数文件 (marker_params.json)", "",
            "JSON文件 (*.json);;所有文件 (*.*)")
        if not params_path:
            QMessageBox.warning(self, "警告", "未选择参数文件，将使用默认参数")
            self.params_json_path = None
        else:
            self.params_json_path = params_path

        # 4. 选择保存ROI和匹配数据的文件夹（通常是图像文件夹下的data子文件夹）
        data_dir = QFileDialog.getExistingDirectory(
            self, "选择data文件夹（包含roi_masks.npz，同级目录应有result文件夹）", "")
        if not data_dir:
            QMessageBox.warning(self, "警告", "未选择数据文件夹，程序将退出")
            QTimer.singleShot(100, self.close)
            return
        self.data_save_dir = data_dir

        # 加载配置
        self.load_all_configs()

    def select_video(self):
        """手动选择视频"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if video_path:
            self.video_path = video_path
            self.load_video()

    def load_all_configs(self):
        """加载所有配置"""
        # 加载标定参数
        if not self.load_calibration():
            return

        # 加载检测器
        self.load_detector()

        # 加载视频
        if not self.load_video():
            return

        # 处理首帧
        self.process_first_frame()

    def load_calibration(self):
        """加载标定参数"""
        try:
            self.stereo_params = self.load_stereo_params(self.calib_dir)
            self.validate_params(self.stereo_params)
            self.calib_loaded = True
            self.status_bar.showMessage(f"标定参数已加载: {self.calib_dir}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标定参数失败: {str(e)}")
            return False

    def load_stereo_params(self, param_dir):
        """加载立体标定参数"""
        params = {}
        params['K1'] = np.loadtxt(os.path.join(param_dir, "K1.txt")).reshape(3, 3).astype(np.float64)
        params['K2'] = np.loadtxt(os.path.join(param_dir, "K2.txt")).reshape(3, 3).astype(np.float64)
        params['D1'] = np.loadtxt(os.path.join(param_dir, "D1.txt")).reshape(1, 5).astype(np.float64)
        params['D2'] = np.loadtxt(os.path.join(param_dir, "D2.txt")).reshape(1, 5).astype(np.float64)
        params['R'] = np.loadtxt(os.path.join(param_dir, "R.txt")).reshape(3, 3).astype(np.float64)
        params['T'] = np.loadtxt(os.path.join(param_dir, "T.txt")).reshape(3, 1).astype(np.float64)
        return params

    def validate_params(self, params):
        """验证标定参数"""
        assert params['K1'].shape == (3, 3), "K1必须是3x3矩阵"
        assert params['K2'].shape == (3, 3), "K2必须是3x3矩阵"
        assert params['D1'].shape == (1, 5), "D1应为5参数"
        assert params['R'].shape == (3, 3), "R必须是3x3旋转矩阵"
        assert params['T'].size == 3, "T应为3维平移向量"

    def load_detector(self):
        """加载检测器"""
        config_path = self.params_json_path if self.params_json_path else "marker_params.json"
        self.detector = CircleDetector(config_path=config_path, verbose=True)

    def load_video(self):
        """加载视频"""
        if not self.video_path:
            return False

        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开视频: {self.video_path}")
            return False

        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 60

        self.slider_progress.setRange(0, self.total_frames - 1)
        self.spin_speed.setValue(int(self.fps))
        self.status_bar.showMessage(f"视频已加载: {self.total_frames}帧, {self.fps:.1f}fps")
        return True

    def process_first_frame(self):
        """处理首帧 - 直接从result文件加载已处理好的首帧数据作为参考"""
        if not self.video_cap or not self.calib_loaded:
            return

        # 读取首帧（仅用于显示）
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_cap.read()
        if not ret:
            QMessageBox.critical(self, "错误", "无法读取视频首帧")
            return

        h, w = frame.shape[:2]
        mirror_axis = w // 2

        # 从matched_points.npz加载mirror_axis
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
            except Exception as e:
                pass

        # 加载ROI（用于后续帧处理）
        roi_file = os.path.join(self.data_save_dir, "roi_masks.npz")
        if os.path.exists(roi_file):
            try:
                with np.load(roi_file) as data:
                    left_mask = data['left_mask']
                    right_mask = data['right_mask']
                    if left_mask.ndim == 3: left_mask = left_mask[:, :, 0]
                    if right_mask.ndim == 3: right_mask = right_mask[:, :, 0]
                    self.drawing['left']['mask'] = left_mask
                    self.drawing['right']['mask'] = right_mask
            except Exception as e:
                QMessageBox.warning(self, "警告", f"ROI加载失败: {e}")
                return
        else:
            QMessageBox.warning(self, "警告", f"未找到ROI文件: {roi_file}")
            return

        # 从result文件夹加载首帧数据（这是V4_test处理PNG后保存的正确数据）
        result_dir = os.path.join(self.data_save_dir, "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")

        if not os.path.exists(first_frame_file):
            QMessageBox.warning(self, "警告", f"未找到首帧数据文件: {first_frame_file}\n请先用V4_test处理图片序列")
            return

        try:
            # 读取首帧数据：X, Y, Z, Left_x, Left_y, Right_x, Right_y
            data = np.genfromtxt(first_frame_file, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 7:
                QMessageBox.warning(self, "警告", "首帧数据文件格式错误")
                return

            # 加载已处理好的数据
            points_3d = data[:, :3].astype(np.float64)
            pts1_R = data[:, 3:5].astype(np.float32)  # 左视图2D坐标
            pts2_R = data[:, 5:7].astype(np.float32)  # 右视图2D坐标

        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取首帧数据失败: {e}")
            return

        # 标定参数
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']

        # 立体校正（用于后续帧处理）
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

        # 对加载的2D坐标进行去畸变
        pts1 = cv2.undistortPoints(pts1_R, K1, D1, R=R1, P=P1).squeeze()
        pts2 = cv2.undistortPoints(pts2_R, K2, D2, R=R2, P=P2).squeeze()

        # 更新全局数据
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
        # 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        # 构建坐标系
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix

        # 显示首帧
        self.current_frame_idx = 0
        self.display_frame(frame, pts1_R, pts2_R)
        self.update_3d_view(points_3d)

        # 启用控制按钮
        self.btn_play_pause.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.status_bar.showMessage(f"首帧处理完成，加载了{len(points_3d)}个标记点")

    def reset_first_frame(self):
        """重置到首帧"""
        self.current_frame_idx = 0

        # 重置参考点
        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()

        if self.input_mode == 'video':
            self.slider_progress.setValue(0)
            # 显示首帧
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame, self.FRAME_DATA['left_points_0_R'], self.FRAME_DATA['right_points_0_R'])
                self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        else:
            # 摄像头模式：仅重置参考点，显示基准点云
            self.update_3d_view(self.FRAME_DATA['base_3d_points'])

        self.status_bar.showMessage("已重置参考点")
        self.consecutive_abnormal_frames = 0

    def set_current_as_base(self):
        """将当前帧设置为新的基准帧"""
        if not self.FRAME_DATA['initialized']:
            self.status_bar.showMessage("请先初始化首帧")
            return

        # 获取当前帧的检测数据
        left_pts = self.FRAME_DATA.get('left_points_0_pre')
        right_pts = self.FRAME_DATA.get('right_points_0_pre')

        if left_pts is None or right_pts is None:
            self.status_bar.showMessage("当前帧数据不可用")
            return

        # 更新基准点数据
        self.FRAME_DATA['left_points_0_R'] = left_pts.copy()
        self.FRAME_DATA['right_points_0_R'] = right_pts.copy()

        # 重新计算基准3D点
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']
        w, h = self.FRAME_DATA['mirror_axis'] * 2, self.FRAME_DATA.get('frame_height', 480)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        l_pts_ud = cv2.undistortPoints(left_pts, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_pts, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

        self.FRAME_DATA['base_3d_points'] = points_3d

        # 更新基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(left_pts.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(right_pts.astype(np.float64))

        # 注意：不重新构建坐标系，保持原始坐标系不变
        # 这样可以确保坐标系始终一致，只更新基准3D点用于计算形变

        # 更新3D视图
        self.update_3d_view(points_3d)

        self.status_bar.showMessage(f"已将当前帧设置为新的基准帧（帧 {self.current_frame_idx}）")
        self.consecutive_abnormal_frames = 0

    def process_frame(self, frame):
        """处理后续帧"""
        if not self.FRAME_DATA['initialized']:
            return None, None, None, None, None
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
            valid_mask_l = min_dists_l <= filter_dist
            left_pts_det = left_pts_det_raw[valid_mask_l]
        else:
            left_pts_det = left_pts_det_raw
        if len(right_pts_det_raw) > 0 and len(pre_r) > 0:
            dist_matrix_r = cdist(right_pts_det_raw, pre_r)
            min_dists_r = np.min(dist_matrix_r, axis=1)
            valid_mask_r = min_dists_r <= filter_dist
            right_pts_det = right_pts_det_raw[valid_mask_r]
        else:
            right_pts_det = right_pts_det_raw
        matched_pairs = self.auto_match_points(left_pts_det, right_pts_det, pre_l, pre_r, self.max_match_dist)
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
                    abnormal_reason.append(f"Z轴深度异常")
                if not is_abnormal and len(points_3d) >= 2:
                    from scipy.spatial.distance import pdist
                    distances = pdist(points_3d)
                    if np.any(distances < 0.3):
                        is_abnormal = True
                        abnormal_reason.append(f"点间距异常")
        else:
            is_abnormal = True
            abnormal_reason = ["灾难性熔断"]
            base_3d = self.FRAME_DATA['base_3d_points']

        # ── 连续异常帧计数与分级处理 ──
        if not is_abnormal:
            self.consecutive_abnormal_frames = 0
            self.FRAME_DATA['left_points_0_pre'] = left_points_R
            self.FRAME_DATA['right_points_0_pre'] = right_points_R
            return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask

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
                    np.ones(n_pts, dtype=bool))
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
                    np.ones(n_pts, dtype=bool))

    def apply_roi_mask(self, points, mask):
        """应用ROI掩膜"""
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
        """线性三角化"""
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
            """使用全局匈牙利算法进行匹配

            Args:
                ref_pts: 前一帧的参考点 (N, 2)
                det_pts: 当前帧检测到的点 (M, 2)
                threshold: 最大匹配距离阈值

            Returns:
                final_map: 字典，key为参考点索引，value为检测点索引
            """
            if len(ref_pts) == 0 or len(det_pts) == 0:
                return {}

            # 计算距离矩阵
            dist_matrix = cdist(ref_pts, det_pts)

            # 创建代价矩阵，超过阈值的设为大值（避免匹配）
            large_value = threshold * 10  # 使用大值代替无穷大，避免数值问题
            cost_matrix = dist_matrix.copy()
            cost_matrix[cost_matrix > threshold] = large_value

            # 处理参考点数量与检测点数量不等的情况
            n_ref = len(ref_pts)
            n_det = len(det_pts)

            if n_ref > n_det:
                # 参考点多于检测点，需要填充虚拟检测点
                padding = np.full((n_ref, n_ref - n_det), large_value)
                cost_matrix = np.hstack([cost_matrix, padding])
            elif n_det > n_ref:
                # 检测点多于参考点，需要填充虚拟参考点
                padding = np.full((n_det - n_ref, n_det), large_value)
                cost_matrix = np.vstack([cost_matrix, padding])

            # 使用匈牙利算法求解全局最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 构建匹配结果，过滤掉超过阈值的匹配
            final_map = {}
            for r_idx, d_idx in zip(row_ind, col_ind):
                # 只保留有效的匹配（在原始矩阵范围内且距离在阈值内）
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

    def build_coordinate_system_pca(self, points_3d):
        """使用PCA构建坐标系"""
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
        """转换到局部坐标系"""
        origin = self.FRAME_DATA['transform_origin']
        rotation = self.FRAME_DATA['transform_rotation']
        translated = points - origin
        return np.dot(translated, rotation)

    def display_frame(self, frame, left_pts=None, right_pts=None, left_lost=None, right_lost=None):
        """显示帧"""
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

        # 转换为Qt图像
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放以适应标签（使用快速缩放）
        label_size = self.lbl_video.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)

        # 仅视频模式更新帧信息（摄像头模式在on_camera_frame中更新）
        if self.input_mode == 'video':
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx} / {self.total_frames - 1}")

    def update_3d_view(self, points_3d):
        """更新3D视图"""
        # 拖拽中或拖拽刚结束时不更新
        if self.is_dragging:
            return
        if time.time() - self.drag_release_time < self.drag_cooldown:
            return

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)
            points[:, 1] = -points[:, 1]  # 沿x=0平面镜像

            # 计算形变
            base_local = self.transform_to_local_coordinates(self.FRAME_DATA['base_3d_points'])
            base_local[:, 1] = -base_local[:, 1]  # 沿x=0平面镜像

            # 计算位移向量
            displacement_vectors = points - base_local
            deformation = np.linalg.norm(displacement_vectors, axis=1)

            # 计算位移统计
            self.calculate_displacement_stats(displacement_vectors, deformation)
            self.current_points_3d = points

            sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=deformation, cmap='jet', s=50, vmin=0, vmax=1.5)
            cbar = self.fig_3d.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label('Deformation (mm)', rotation=270, labelpad=15)

            # 绘制平均位移箭头
            if self.avg_displacement_magnitude > 0.01:
                center = np.mean(points, axis=0)
                arrow_scale = 5.0  # 箭头缩放因子
                ax.quiver(center[0], center[1], center[2],
                         self.avg_displacement_vector[0] * arrow_scale,
                         self.avg_displacement_vector[1] * arrow_scale,
                         self.avg_displacement_vector[2] * arrow_scale,
                         color='red', arrow_length_ratio=0.3, linewidth=2.5,
                         label=f'Avg: {self.avg_displacement_magnitude:.3f}mm')
                ax.legend(loc='upper left', fontsize=8)
        else:
            points = points_3d.copy()
            points[:, 1] = -points[:, 1]  # 沿x=0平面镜像
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)
            # 重置位移统计
            self.reset_displacement_stats()

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Frame {self.current_frame_idx}')

        # 设置坐标轴比例
        if len(points) > 0:
            ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])

        # 设置视角
        if self.view_saved:
            try:
                ax.view_init(elev=self.view_elev, azim=self.view_azim, roll=self.view_roll)
            except TypeError:
                # 旧版本matplotlib不支持roll参数
                ax.view_init(elev=self.view_elev, azim=self.view_azim)
        else:
            try:
                ax.view_init(elev=90, azim=0, roll=-90)
            except TypeError:
                ax.view_init(elev=90, azim=0)

        self.ax_3d = ax
        self.canvas_3d.draw()

    def calculate_displacement_stats(self, displacement_vectors, deformation):
        """计算位移统计并更新UI显示"""
        if len(displacement_vectors) == 0:
            self.reset_displacement_stats()
            return

        # 计算平均位移向量
        self.avg_displacement_vector = np.mean(displacement_vectors, axis=0)

        # 计算平均位移大小（所有点位移大小的平均值）
        self.avg_displacement_magnitude = np.mean(deformation)

        # 计算最大和最小位移
        self.max_displacement = np.max(deformation)
        self.min_displacement = np.min(deformation)

        # 更新UI显示
        self.update_displacement_ui()

    def reset_displacement_stats(self):
        """重置位移统计数据"""
        self.avg_displacement_vector = np.array([0.0, 0.0, 0.0])
        self.avg_displacement_magnitude = 0.0
        self.max_displacement = 0.0
        self.min_displacement = 0.0
        self.update_displacement_ui()

    def update_displacement_ui(self):
        """更新位移统计UI显示（UI组件已移除，此函数保留为空）"""
        pass

    def update_max_dist(self, value):
        """更新最大匹配距离"""
        self.max_match_dist = value

    # ---- 播放控制 ----
    def toggle_play_pause(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """开始播放"""
        self.is_playing = True
        self.btn_play_pause.setText("暂停")

        # 记录视觉开始时间（用于与触觉同步）
        if self.visual_start_time is None:
            self.visual_start_time = time.time()

        if self.input_mode == 'video':
            interval = int(1000 / self.spin_speed.value())
            self.play_timer.start(interval)
        self.status_bar.showMessage("播放中...")

    def pause_playback(self):
        """暂停播放"""
        self.is_playing = False
        self.btn_play_pause.setText("播放")
        if self.input_mode == 'video':
            self.play_timer.stop()
        self.status_bar.showMessage("已暂停")

    def update_playback_speed(self, fps):
        """更新播放速度"""
        if self.input_mode == 'video' and self.is_playing:
            interval = int(1000 / fps)
            self.play_timer.setInterval(interval)
        elif self.input_mode == 'camera' and self.camera_thread is not None:
            self.camera_thread.target_fps = fps

    def on_timer_tick(self):
        """定时器触发（仅视频模式）"""
        if self.input_mode != 'video':
            return
        if self.current_frame_idx >= self.total_frames - 1:
            self.current_frame_idx = 0
            # 重置参考点
            self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
            self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.video_cap.read()
        if not ret:
            self.pause_playback()
            return

        if self.current_frame_idx == 0:
            # 首帧直接显示
            self.display_frame(frame, self.FRAME_DATA['left_points_0_R'], self.FRAME_DATA['right_points_0_R'])
            self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        else:
            # 处理后续帧
            points_3d, left_pts, right_pts, left_lost, right_lost = self.process_frame(frame)
            if points_3d is not None:
                self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)
                self.update_3d_view(points_3d)

        self.slider_progress.blockSignals(True)
        self.slider_progress.setValue(self.current_frame_idx)
        self.slider_progress.blockSignals(False)

        self.current_frame_idx += 1

    # ---- 进度条控制（仅视频模式） ----
    def on_slider_pressed(self):
        """滑块按下"""
        if self.input_mode != 'video':
            return
        self.slider_dragging = True
        if self.is_playing:
            self.play_timer.stop()

    def on_slider_released(self):
        """滑块释放"""
        if self.input_mode != 'video':
            return
        self.slider_dragging = False
        self.seek_to_frame(self.slider_progress.value())
        if self.is_playing:
            interval = int(1000 / self.spin_speed.value())
            self.play_timer.start(interval)

    def on_slider_changed(self, value):
        """滑块值改变"""
        if self.input_mode != 'video':
            return
        if self.slider_dragging:
            self.lbl_frame_info.setText(f"帧: {value} / {self.total_frames - 1}")

    def seek_to_frame(self, frame_idx):
        """跳转到指定帧（仅视频模式）"""
        if self.input_mode != 'video':
            return
        # 重置参考点到首帧
        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()

        # 从头开始逐帧处理到目标帧
        for i in range(frame_idx + 1):
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video_cap.read()
            if not ret:
                break

            if i == 0:
                left_pts = self.FRAME_DATA['left_points_0_R']
                right_pts = self.FRAME_DATA['right_points_0_R']
                points_3d = self.FRAME_DATA['base_3d_points']
                left_lost = right_lost = None
            else:
                points_3d, left_pts, right_pts, left_lost, right_lost = self.process_frame(frame)

        self.current_frame_idx = frame_idx
        if points_3d is not None:
            self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)
            self.update_3d_view(points_3d)

    # ---- 3D视图鼠标事件 ----
    def on_3d_mouse_press(self, event):
        """3D视图鼠标按下"""
        self.is_dragging = True
        # 记录拖拽前的播放状态
        self._was_playing_before_drag = self.is_playing
        # 暂停播放（视频模式和摄像头模式都暂停）
        if self.is_playing:
            if self.input_mode == 'video':
                self.play_timer.stop()
            # 摄像头模式：设置is_playing为False暂停3D更新
            self.is_playing = False

    def on_3d_mouse_motion(self, event):
        """3D视图鼠标移动 - 实时更新视角参数"""
        if self.is_dragging and hasattr(self, 'ax_3d'):
            self.view_elev = self.ax_3d.elev
            self.view_azim = self.ax_3d.azim
            self.view_roll = getattr(self.ax_3d, 'roll', 0)  # matplotlib 3.6+ 支持roll
            self.view_saved = True

    def on_3d_mouse_release(self, event):
        """3D视图鼠标释放"""
        if self.is_dragging:
            self.is_dragging = False
            self.drag_release_time = time.time()  # 记录释放时间
            if hasattr(self, 'ax_3d'):
                self.view_elev = self.ax_3d.elev
                self.view_azim = self.ax_3d.azim
                self.view_roll = getattr(self.ax_3d, 'roll', 0)  # 保存roll参数
                self.view_saved = True

            # 恢复拖拽前的播放状态
            if self._was_playing_before_drag:
                self.is_playing = True
                if self.input_mode == 'video':
                    interval = int(1000 / self.spin_speed.value())
                    self.play_timer.start(interval)
                self._was_playing_before_drag = False

    # ---- 触觉传感器相关方法 ----
    def get_serial_ports(self):
        """获取可用的串口端口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def refresh_serial_ports(self):
        """刷新串口列表"""
        self.cbb_serial_port.clear()
        self.cbb_serial_port.addItems(self.get_serial_ports())

    def start_serial_reading(self):
        """启动串口数据读取"""
        port = self.cbb_serial_port.currentText()
        if not port:
            QMessageBox.warning(self, "警告", "请选择串口")
            return
        baudrate = int(self.cbb_serial_baud.currentText())

        # 清空数据
        self.data_ptac_adc1 = []
        self.data_ptac_adc2 = []
        self.data_ptac_adc3 = []
        self.data_ptac_adc4 = []
        self.data_ptac_adc5 = []
        self.tactile_timestamps = []

        # 记录开始时间（与视觉同步）
        self.tactile_start_time = time.time()
        if self.visual_start_time is None:
            self.visual_start_time = self.tactile_start_time

        if self.serial_receiver is None or not self.serial_receiver.isRunning():
            self.serial_receiver = SerialReceiver(port, baudrate)
            self.serial_receiver.data_received.connect(self.update_tactile_data)
            self.serial_receiver.start()
            self.status_bar.showMessage(f"串口已连接: {port}")

    def stop_serial_reading(self):
        """停止串口数据读取"""
        if self.serial_receiver and self.serial_receiver.isRunning():
            self.serial_receiver.stop()
            self.status_bar.showMessage("串口已断开")

    def update_tactile_data(self, data_frame_datas, timestamp):
        """更新触觉传感器数据"""
        flag_adc = data_frame_datas[0]
        data_frame = data_frame_datas[1:]

        REFERENCE_VOLTAGE = 3.3
        MAX_VALUE = 2 ** 23

        decimal_data = []
        for i in range(0, len(data_frame), 3):
            value = data_frame[i:i + 3]
            decimal_value = self.bytes_to_decimal(value)
            voltage = (decimal_value / MAX_VALUE) * REFERENCE_VOLTAGE
            decimal_data.append(voltage)

        # 记录时间戳
        self.tactile_timestamps.append(timestamp)

        if flag_adc == 0:
            self.data_ptac_adc1.append(decimal_data)
        elif flag_adc == 1:
            self.data_ptac_adc2.append(decimal_data)
        elif flag_adc == 2:
            self.data_ptac_adc3.append(decimal_data)
        elif flag_adc == 3:
            self.data_ptac_adc4.append(decimal_data)
        elif flag_adc == 4:
            self.data_ptac_adc5.append(decimal_data)

    def bytes_to_decimal(self, data):
        """将字节数据转换为十进制"""
        byte1, byte2, byte3 = data
        adc_value = (byte1 << 16) | (byte2 << 8) | byte3
        if adc_value & 0x800000:
            adc_value -= 0x1000000
        return adc_value

    def update_tactile_plot(self):
        """更新触觉信号图表"""
        group_index = self.cbb_adc_group.currentIndex()
        if group_index == 0:
            data_adc_i = self.data_ptac_adc1
        elif group_index == 1:
            data_adc_i = self.data_ptac_adc2
        elif group_index == 2:
            data_adc_i = self.data_ptac_adc3
        elif group_index == 3:
            data_adc_i = self.data_ptac_adc4
        elif group_index == 4:
            data_adc_i = self.data_ptac_adc5
        else:
            data_adc_i = []

        if not data_adc_i:
            return

        # 获取用户选择的通道（1-8，转换为索引0-7）
        selected_channels = [
            self.spin_ch1.value() - 1,
            self.spin_ch2.value() - 1,
            self.spin_ch3.value() - 1
        ]

        # 仅绘制最近的 2000 个点
        display_data = data_adc_i[-2000:]
        num_points = len(display_data)
        # 使用连续整数索引作为x轴，让最新点在x=0处
        x_axis = list(range(-num_points + 1, 1))

        for i, curve in enumerate(self.tactile_curves):
            ch_idx = selected_channels[i]  # 获取用户选择的通道索引
            # 更新图表标题
            self.tactile_plots[i].setTitle(f"CH:{ch_idx + 1}", color='k', size='10pt')
            # 获取对应通道的数据
            y_data = [row[ch_idx] for row in display_data if len(row) > ch_idx]
            if len(y_data) > 0:
                curve.setData(x_axis[:len(y_data)], y_data)

    def closeEvent(self, event):
        """关闭窗口"""
        # 停止触觉传感器
        if self.serial_receiver is not None:
            self.serial_receiver.stop()
        self.tactile_timer.stop()

        if self.process_thread is not None:
            self.process_thread.stop()
        if self.camera_thread is not None:
            self.camera_thread.stop()
        if self.video_cap:
            self.video_cap.release()
        self.play_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPointCloudPlayer()
    window.show()
    sys.exit(app.exec_())
