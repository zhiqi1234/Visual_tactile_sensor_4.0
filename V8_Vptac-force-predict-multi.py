# -*- coding: utf-8 -*-
'''
V8_Vptac-force-predict-multi — 3通道视触觉传感器同时采集
功能说明：
1. 支持3路USB摄像头同时实时采集，逐帧进行标记点检测和三维重建
2. 启动前为每个传感器选择：标定文件夹、参数JSON文件、data文件夹、力预测模型目录
3. GUI标签页分"传感器数据"(3路点云+压电+力曲线)和"视频画面"(3路视频+标记)
4. 六维力传感器实时订阅与显示（3传感器共享同一物理传感器）
5. 力预测模型实时推理并显示（每传感器独立模型）
6. 统一HDF5采集：/sensor_N/vision, /force, /reference, /meta
7. 压电传感器：1个COM口，3传感器独立选择ADC组和通道
'''
import sys
import signal
import os
from collections import deque
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
import serial
import serial.tools.list_ports
import topic  # type: ignore
import message  # type: ignore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QGroupBox, QStatusBar, QSpinBox, QSplitter, QCheckBox,
                             QComboBox, QTabWidget, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import linear_sum_assignment
import pyqtgraph as pg

# ForcePredictor 根据模型类型动态导入（见 load_force_predictor）

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────── 压电串口采集线程（V8修改版：提取全部8通道）───────────────────────

class PiezoSerialThread(QThread):
    """压电传感器串口采集线程 — V8修改版，提取全部8通道数据"""
    _FRAME_LEN = 29
    _REF_V = 3.3
    _MAX_VAL = 2 ** 23
    _N_GROUPS = 5
    _N_CHANNELS = 8

    def __init__(self, port='COM1', baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.buffer = b''
        self.last_clear_time = 0

        # 按ADC组(0-4) × 通道(0-7) 分组的共享缓冲区
        self._group_bufs = [[deque(maxlen=6000) for _ in range(self._N_CHANNELS)] for _ in range(self._N_GROUPS)]
        self._buf_lock = threading.Lock()

    def get_buffer_snapshot(self, adc_group, channel):
        """获取指定ADC组和通道的波形数据"""
        with self._buf_lock:
            return list(self._group_bufs[adc_group][channel])

    def get_window_values(self, adc_group, channel, t_start, t_end):
        """获取指定ADC组和通道时间窗口内的电压值"""
        with self._buf_lock:
            result = []
            for t, v in reversed(self._group_bufs[adc_group][channel]):
                if t < t_start:
                    break
                if t <= t_end:
                    result.append(v)
        return result

    def get_buffer_size(self, adc_group, channel):
        """获取指定缓冲区当前帧数"""
        with self._buf_lock:
            return len(self._group_bufs[adc_group][channel])

    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial.reset_input_buffer()
            self.running = True
            self.last_clear_time = time.time()

            while self.running:
                if self.serial.in_waiting > 0:
                    self.buffer += self.serial.read(self.serial.in_waiting)

                    current_time = time.time()
                    if len(self.buffer) > 5000 or (current_time - self.last_clear_time > 2.0 and len(self.buffer) > 1000):
                        self.buffer = self.buffer[-1000:]
                        self.last_clear_time = current_time

                    self.process_data()
                else:
                    time.sleep(0.0005)
        except serial.SerialException as e:
            print(f"压电串口错误: {e}")

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
                break

            if aaaa_index + FRAME_LENGTH <= len(self.buffer):
                if self.buffer[aaaa_index + FRAME_LENGTH - 2: aaaa_index + FRAME_LENGTH] == b'\xff\xff':
                    group_data = self.buffer[aaaa_index + 2: aaaa_index + FRAME_LENGTH - 2]
                    flag_adc = group_data[0]
                    data_frame = group_data[1:]

                    if flag_adc < self._N_GROUPS:
                        ts = time.time()
                        with self._buf_lock:
                            for ch in range(self._N_CHANNELS):
                                offset = ch * 3
                                if offset + 3 <= len(data_frame):
                                    raw = data_frame[offset:offset + 3]
                                    decimal_value = self.bytes_to_decimal(raw)
                                    voltage = (decimal_value / self._MAX_VAL) * self._REF_V
                                    if ch == 0 or ch == 1:
                                        voltage = -voltage
                                    self._group_bufs[flag_adc][ch].append((ts, voltage))

                    self.buffer = self.buffer[aaaa_index + FRAME_LENGTH:]
                else:
                    self.buffer = self.buffer[aaaa_index + 2:]
            else:
                break

    def bytes_to_decimal(self, data):
        byte1, byte2, byte3 = data
        adc_value = (byte1 << 16) | (byte2 << 8) | byte3
        if adc_value & 0x800000:
            adc_value -= 0x1000000
        return adc_value


# ─────────────────────── 相机线程（V7原样）───────────────────────

class CameraThread(QThread):
    """摄像头采集线程"""
    frame_signal = pyqtSignal(np.ndarray, float)
    fps_signal = pyqtSignal(float)

    def __init__(self, camera_index=0, fps=30, rotate_180=True):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = fps
        self.rotate_180 = rotate_180
        self.running = True
        self.video_cap = None

    def run(self):
        self.video_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.video_cap.isOpened():
            return
        self.video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        actual_w = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fourcc = int(self.video_cap.get(cv2.CAP_PROP_FOURCC))
        print(f"[Camera {self.camera_index}] 分辨率={actual_w}x{actual_h}, FOURCC={actual_fourcc:08x}, "
              f"目标FPS={self.target_fps}")
        frame_interval = 1.0 / self.target_fps

        fps_count = 0
        fps_last_time = time.time()

        while self.running:
            start_time = time.time()
            ret, frame = self.video_cap.read()
            if ret:
                capture_timestamp = time.time()
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.frame_signal.emit(frame, capture_timestamp)

                fps_count += 1
                now = time.time()
                if now - fps_last_time >= 0.5:
                    fps = fps_count / (now - fps_last_time)
                    self.fps_signal.emit(fps)
                    fps_count = 0
                    fps_last_time = now

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


# ─────────────────────── 帧处理线程（V7原样）───────────────────────

class FrameProcessThread(QThread):
    """帧处理线程 - 将耗时的检测和重建放到后台"""
    result_signal = pyqtSignal(object, object, object, object, object, object, object, object, object)

    def __init__(self):
        super().__init__()
        self.running = True
        self.frame_queue = []
        self.lock = threading.Lock()
        self.process_func = None
        self.render_func = None
        self.has_new_frame = threading.Event()

    def set_processor(self, process_func):
        self.process_func = process_func

    def set_renderer(self, render_func):
        self.render_func = render_func

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
                    qt_image = self.render_func(frame, left_pts, right_pts, left_lost, right_lost) if self.render_func else None
                    self.result_signal.emit(qt_image, points_3d, left_pts, right_pts,
                                            left_lost, right_lost, timestamp, is_abnormal, None)
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


# ─────────────────────── 圆检测器（V7原样）───────────────────────

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
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

        pairs = tree.query_pairs(min_distance)

        to_remove = set()
        for i, j in pairs:
            to_remove.add(j)

        result = [points[i] for i in range(len(points)) if i not in to_remove]
        return result

    def detect(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = self._clahe.apply(gray)

        b_val = max(1, self.params.get("blur", 3))
        if b_val % 2 == 0: b_val += 1
        b_val = min(b_val, 7)
        blur = cv2.GaussianBlur(gray, (b_val, b_val), 0)

        points = self._detect_by_local_minima(gray, blur)

        min_area = self.params["min_area"]
        avg_radius = int(np.sqrt(min_area / np.pi))
        dedup_radius = max(8, avg_radius * 2.5)
        points = self._remove_duplicate_points(points, dedup_radius)

        return points

# ─────────────────────── 主窗口 ───────────────────────

class V8MainWindow(QMainWindow):
    """V8 主窗口: 3通道点云显示 + 力预测 + 统一HDF5采集"""

    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_input_config_v8.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("V8 - 3通道视触觉传感器采集")
        self.setGeometry(50, 50, 1280, 900)

        # 项目根目录
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        # 全局设置
        self.fps = 30
        self.max_match_dist = 50

        # 当前激活的数据显示页 (0=传感器数据, 1=视频画面)
        self.active_display_tab = 0

        # 六维力传感器（3传感器共享）
        self.ft_node = None
        self.ft_subscription = None
        self.ft_lock = threading.Lock()
        self.latest_ft_data = None
        self.latest_ft_timestamp = 0.0
        self.ft_bias = np.zeros(6, dtype=np.float64)
        self.ft_recent_buffer = deque(maxlen=100)

        # 压电传感器（1个COM口，3传感器共享线程）
        self.piezo_thread = None
        self.piezo_port = None

        # 力预测器（每传感器独立加载）
        # self.sensors[i]["force_predictor"]

        # 播放控制
        self.is_playing = False

        # HDF5回放
        self.h5_file = None
        self.h5_mode = False

        # 初始化3个传感器状态
        self.sensors = []
        self._sensor_widgets = []  # GUI控件引用
        for i in range(3):
            s = self._make_sensor_state(i)
            self.sensors.append(s)
            self._sensor_widgets.append({})

        # 录制缓冲（3路视觉 + 1路共享力传感器）
        self.is_recording = False
        self.rec_vision = [
            {"timestamps": [], "xyz": [], "abnormal": [], "predicted_force": []}
            for _ in range(3)
        ]
        self.rec_force = {"timestamps": [], "ft_values": []}

        # 初始化界面
        self.init_ui()

        # 启动六维力传感器订阅
        self.start_ft_subscription()

        # 压电波形刷新定时器 (10Hz)
        self._piezo_timer = QTimer(self)
        self._piezo_timer.timeout.connect(self._update_all_piezo_plots)
        self._piezo_timer.start(100)

        # 自动加载上次配置
        QTimer.singleShot(100, self.auto_load_last_config)

    def _make_sensor_state(self, idx):
        """创建单个传感器的初始状态字典"""
        return {
            "calib_dir": None,
            "params_json_path": None,
            "data_save_dir": None,
            "model_dir": None,
            "camera_index": idx,
            "rotate_180": idx != 0,
            "piezo_adc_group": idx,
            "piezo_channel": idx * 2,
            "enabled": False,
            "camera_thread": None,
            "process_thread": None,
            "latest_camera_frame": None,
            "current_frame_idx": 0,
            "consecutive_abnormal_frames": 0,
            "pca_flip_xy": False,
            "stereo_params": None,
            "calib_loaded": False,
            "detector": None,
            "force_predictor": None,
            "latest_predicted_force": None,
            "piezo_window_ms": 33,
            "FRAME_DATA": {
                "initialized": False,
                "roi_masks": None,
                "P1": None, "P2": None,
                "left_points_0": None, "right_points_0": None,
                "left_points_0_R": None, "right_points_0_R": None,
                "left_points_0_pre": None, "right_points_0_pre": None,
                "base_3d_points": None,
                "mirror_axis": None,
                "transform_origin": None,
                "transform_rotation": None,
                "left_edges": None, "right_edges": None,
                "current_left_crossings": [], "current_right_crossings": [],
            },
            "drawing": {"left": {"mask": None}, "right": {"mask": None}},
            "_left_img_buf": None, "_right_img_buf": None,
            "_pre_l_tree": None, "_pre_r_tree": None,
            "_pre_l_tree_id": None, "_pre_r_tree_id": None,
            "_stereo_rectify_cache": {},
            "_scatter_plot": None, "_colorbar": None, "_force_arrow_plot": None,
            "view_elev": 90, "view_azim": 0, "view_roll": -90,
            "view_saved": False, "drag_release_time": 0,
            "last_3d_update_time": 0, "pointcloud_frame_counter": 0,
            "is_dragging": False, "_was_playing_before_drag": False,
            "min_3d_update_interval": 0.1, "pointcloud_update_interval": 3,
            "force_history_len": 300,
            "force_actual_history": np.zeros((300, 6)),
            "force_pred_history": np.zeros((300, 6)),
            "force_history_idx": 0,
            "_force_data_min": np.zeros(6), "_force_data_max": np.zeros(6),
            "_force_ylims": None, "_force_plot_ready": False,
            "_force_backgrounds": None, "_force_lines_actual": None,
            "_force_lines_pred": None, "_force_vlines": None,
            "_force_title_counter": 0,
        }


    # ──────────────────── UI ────────────────────

    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── 上部控制栏 ──
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(2, 2, 2, 2)

        # 压电区域
        lbl_piezo = QLabel("压电:")
        self.cbb_piezo_port = QComboBox()
        self.cbb_piezo_port.addItems(self._get_serial_ports())
        self.cbb_piezo_port.setMinimumWidth(80)
        self.cbb_piezo_port.setToolTip("选择压电蓝牙串口")
        self.btn_piezo_connect = QPushButton("连接压电")
        self.btn_piezo_connect.clicked.connect(self.toggle_piezo_connection)

        ctrl_layout.addWidget(lbl_piezo)
        ctrl_layout.addWidget(self.cbb_piezo_port)
        ctrl_layout.addWidget(self.btn_piezo_connect)
        ctrl_layout.addWidget(self._vsep())

        # 操作按钮
        self.btn_record = QPushButton("开始采集")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        self.btn_load_h5 = QPushButton("加载HDF5")
        self.btn_load_h5.clicked.connect(self.load_h5_file)

        ctrl_layout.addWidget(self.btn_record)
        ctrl_layout.addWidget(self.btn_load_h5)
        ctrl_layout.addWidget(self._vsep())

        # 速度和匹配距离
        lbl_speed = QLabel("速度:")
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 30)
        self.spin_fps.setValue(30)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.valueChanged.connect(self.update_playback_speed)

        lbl_match = QLabel("匹配:")
        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(1, 500)
        self.spin_max_dist.setValue(50)
        self.spin_max_dist.setSuffix(" px")
        self.spin_max_dist.valueChanged.connect(lambda v: setattr(self, 'max_match_dist', v))

        ctrl_layout.addWidget(lbl_speed)
        ctrl_layout.addWidget(self.spin_fps)
        ctrl_layout.addWidget(lbl_match)
        ctrl_layout.addWidget(self.spin_max_dist)
        ctrl_layout.addWidget(self._vsep())

        # 力传感器
        self.btn_ft_auto_zero = QPushButton("自动调零")
        self.btn_ft_auto_zero.clicked.connect(self.auto_zero_ft)
        self.btn_ft_reset_zero = QPushButton("重置调零")
        self.btn_ft_reset_zero.clicked.connect(self.reset_zero_ft)
        self.chk_show_actual = QCheckBox("显示实际值")
        self.chk_show_actual.setChecked(True)
        self.chk_show_actual.stateChanged.connect(self.on_show_actual_changed)

        ctrl_layout.addWidget(self.btn_ft_auto_zero)
        ctrl_layout.addWidget(self.btn_ft_reset_zero)
        ctrl_layout.addWidget(self.chk_show_actual)

        ctrl_layout.addStretch()

        # 帧信息和FPS
        self.lbl_frame_info = QLabel("帧: 0")
        self._fps_labels = []
        for i in range(3):
            lbl = QLabel(f"Cam{i+1}: --")
            self._fps_labels.append(lbl)
            ctrl_layout.addWidget(lbl)

        ctrl_layout.addWidget(self.lbl_frame_info)

        main_layout.addWidget(ctrl_widget)

        # ── 传感器配置 QTabWidget ──
        self.config_tabs = QTabWidget()
        self.config_tabs.setMaximumHeight(110)
        self._sensor_path_labels = [None, None, None]
        for i in range(3):
            tab = self._make_sensor_config_tab(i)
            self.config_tabs.addTab(tab, f"Sensor {i+1}")
        main_layout.addWidget(self.config_tabs)

        # ── 主显示 QTabWidget ──
        self.display_tabs = QTabWidget()
        self.display_tabs.currentChanged.connect(self._on_display_tab_changed)

        self._sensor_data_tab = self._make_sensor_data_tab()
        self._video_tab = self._make_video_tab()

        self.display_tabs.addTab(self._sensor_data_tab, "传感器数据")
        self.display_tabs.addTab(self._video_tab, "视频画面")
        main_layout.addWidget(self.display_tabs, 1)

        # ── 状态栏 ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请配置各传感器并选择输入源")

        # 力曲线定时器 (3路, 200ms)
        self.force_plot_timer = QTimer(self)
        self.force_plot_timer.timeout.connect(self.update_all_force_plots)
        self.force_plot_timer.start(333)

    @staticmethod
    def _vsep():
        sep = QLabel("|")
        sep.setStyleSheet("color: #aaa; font-size: 14px; padding: 0 1px;")
        sep.setFixedWidth(8)
        return sep

    def _make_sensor_config_tab(self, idx):
        s = self.sensors[idx]
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_select = QPushButton("选择输入源")
        btn_select.clicked.connect(lambda checked, i=idx: self.select_input(i))
        layout.addWidget(btn_select)
        layout.addWidget(QLabel("|"))

        cam_group = QGroupBox("摄像头")
        cam_layout = QHBoxLayout(cam_group)
        cam_layout.addWidget(QLabel("ID:"))
        spin_cam = QSpinBox()
        spin_cam.setRange(0, 10)
        spin_cam.setValue(s['camera_index'])
        spin_cam.valueChanged.connect(lambda v, i=idx: self.sensors[i].update({'camera_index': v}))
        cam_layout.addWidget(spin_cam)
        chk_rotate = QCheckBox("旋转180°")
        chk_rotate.setChecked(s['rotate_180'])
        chk_rotate.stateChanged.connect(lambda state, i=idx: self.sensors[i].update({'rotate_180': bool(state)}))
        cam_layout.addWidget(chk_rotate)
        layout.addWidget(cam_group)
        layout.addWidget(QLabel("|"))

        piezo_group = QGroupBox("压电")
        piezo_layout = QHBoxLayout(piezo_group)
        piezo_layout.addWidget(QLabel("ADC:"))
        cbb_adc = QComboBox()
        cbb_adc.addItems([f"ADC{i+1}" for i in range(5)])
        cbb_adc.setCurrentIndex(s['piezo_adc_group'])
        cbb_adc.currentIndexChanged.connect(lambda v, i=idx: self._on_piezo_adc_changed(i, v))
        piezo_layout.addWidget(cbb_adc)
        piezo_layout.addWidget(QLabel("CH:"))
        cbb_ch = QComboBox()
        cbb_ch.addItems([f"CH{i+1}" for i in range(8)])
        cbb_ch.setCurrentIndex(s['piezo_channel'])
        cbb_ch.currentIndexChanged.connect(lambda v, i=idx: self._on_piezo_ch_changed(i, v))
        piezo_layout.addWidget(cbb_ch)
        layout.addWidget(piezo_group)
        layout.addWidget(QLabel("|"))

        path_group = QGroupBox("路径")
        path_layout = QVBoxLayout(path_group)
        lbl_path = QLabel("未设置")
        lbl_path.setStyleSheet("color: #888; font-size: 8pt;")
        lbl_path.setWordWrap(True)
        self._sensor_path_labels[idx] = lbl_path
        path_layout.addWidget(lbl_path)
        layout.addWidget(path_group, 1)

        self._sensor_widgets[idx] = {
            'btn_select': btn_select, 'spin_cam': spin_cam,
            'chk_rotate': chk_rotate, 'cbb_adc': cbb_adc,
            'cbb_ch': cbb_ch, 'lbl_path': lbl_path,
        }
        return w

    def _make_sensor_data_tab(self):
        tab = QWidget()
        splitter = QSplitter(Qt.Horizontal)

        for i in range(3):
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(2, 2, 2, 2)
            col_layout.setSpacing(2)

            header = QHBoxLayout()
            lbl_title = QLabel(f"<b>Sensor {i+1}</b>")
            header.addWidget(lbl_title)
            header.addStretch()
            btn_reset = QPushButton("重置首帧")
            btn_reset.setMaximumWidth(65)
            btn_reset.setStyleSheet("font-size: 8pt; padding: 2px 4px;")
            btn_reset.clicked.connect(lambda checked, idx=i: self.reset_first_frame(idx))
            header.addWidget(btn_reset)
            btn_set_base = QPushButton("设为首帧")
            btn_set_base.setMaximumWidth(65)
            btn_set_base.setStyleSheet("font-size: 8pt; padding: 2px 4px;")
            btn_set_base.clicked.connect(lambda checked, idx=i: self.set_current_as_base(idx))
            header.addWidget(btn_set_base)
            col_layout.addLayout(header)

            fig_3d = plt.figure(figsize=(3.2, 2.3))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            canvas_3d = FigureCanvas(fig_3d)
            canvas_3d.setMinimumHeight(180)
            canvas_3d.mpl_connect('button_press_event', lambda event, idx=i: self.on_3d_mouse_press(idx, event))
            canvas_3d.mpl_connect('button_release_event', lambda event, idx=i: self.on_3d_mouse_release(idx, event))
            canvas_3d.mpl_connect('motion_notify_event', lambda event, idx=i: self.on_3d_mouse_motion(idx, event))
            col_layout.addWidget(canvas_3d)

            self.sensors[i]['fig_3d'] = fig_3d
            self.sensors[i]['ax_3d'] = ax_3d
            self.sensors[i]['canvas_3d'] = canvas_3d
            self.sensors[i]['btn_reset'] = btn_reset
            self.sensors[i]['btn_set_base'] = btn_set_base

            piezo_widget = pg.GraphicsLayoutWidget()
            piezo_widget.setBackground('w')
            piezo_widget.setMaximumHeight(100)
            piezo_item = piezo_widget.addPlot()
            piezo_item.setLabel('left', 'V')
            piezo_item.setYRange(-3.3, 3.3)
            piezo_item.showGrid(x=True, y=True, alpha=0.3)
            piezo_curve = piezo_item.plot(pen=pg.mkPen('#2196F3', width=2))
            piezo_status = pg.LabelItem(text='未连接', color='#666666', size='9pt')
            piezo_widget.addItem(piezo_status, row=0, col=1)
            col_layout.addWidget(piezo_widget)

            self.sensors[i]['piezo_plot_widget'] = piezo_widget
            self.sensors[i]['piezo_plot_item'] = piezo_item
            self.sensors[i]['piezo_curve_preview'] = piezo_curve
            self.sensors[i]['piezo_status_label'] = piezo_status

            fig_force, axes_force = plt.subplots(2, 3, figsize=(3.2, 2.3))
            canvas_force = FigureCanvas(fig_force)
            canvas_force.setMinimumHeight(180)
            col_layout.addWidget(canvas_force)

            self.sensors[i]['fig_force'] = fig_force
            self.sensors[i]['axes_force'] = axes_force
            self.sensors[i]['canvas_force'] = canvas_force

            splitter.addWidget(col)

        splitter.setSizes([420, 420, 420])
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(splitter)
        return tab

    def _make_video_tab(self):
        tab = QWidget()
        splitter = QSplitter(Qt.Horizontal)

        for i in range(3):
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(2, 2, 2, 2)

            lbl_title = QLabel(f"<b>Sensor {i+1}</b>")
            lbl_title.setAlignment(Qt.AlignCenter)
            col_layout.addWidget(lbl_title)

            lbl_video = QLabel(f"视频显示区域\\nSensor {i+1}")
            lbl_video.setAlignment(Qt.AlignCenter)
            lbl_video.setMinimumSize(320, 240)
            lbl_video.setStyleSheet("background-color: #2a2a2a; color: white;")
            col_layout.addWidget(lbl_video, 1)

            self.sensors[i]['lbl_video'] = lbl_video
            splitter.addWidget(col)

        splitter.setSizes([420, 420, 420])
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(splitter)
        return tab

    def _on_display_tab_changed(self, index):
        self.active_display_tab = index

    # ──────────────────── 串口工具 ────────────────────

    def _get_serial_ports(self):
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception:
            return []

    # ──────────────────── 压电传感器 ────────────────────

    def _on_piezo_adc_changed(self, sensor_idx, value):
        self.sensors[sensor_idx]['piezo_adc_group'] = value

    def _on_piezo_ch_changed(self, sensor_idx, value):
        self.sensors[sensor_idx]['piezo_channel'] = value

    def toggle_piezo_connection(self):
        if self.piezo_thread is None or not self.piezo_thread.isRunning():
            port = self.cbb_piezo_port.currentText()
            if not port:
                QMessageBox.warning(self, "警告", "请选择压电串口")
                return
            self.piezo_port = port
            self.piezo_thread = PiezoSerialThread(port, 921600)
            self.piezo_thread.start()
            self.btn_piezo_connect.setText("断开压电")
            self.btn_piezo_connect.setStyleSheet("background-color: #ff7043; color: white;")
            self.status_bar.showMessage(f"压电传感器已连接: {port}")
            for i in range(3):
                s = self.sensors[i]
                adc = s['piezo_adc_group']
                ch = s['piezo_channel']
                s['piezo_status_label'].setText(f'ADC{adc+1} CH{ch+1}')
                s['piezo_plot_item'].setTitle(f'S{i+1} — ADC{adc+1} CH{ch+1}')
        else:
            self.piezo_thread.stop()
            self.piezo_thread = None
            self.piezo_port = None
            self.btn_piezo_connect.setText("连接压电")
            self.btn_piezo_connect.setStyleSheet("")
            for i in range(3):
                self.sensors[i]['piezo_status_label'].setText('未连接')
                self.sensors[i]['piezo_curve_preview'].setData([])
            self.status_bar.showMessage("压电传感器已断开")

    def _update_all_piezo_plots(self):
        if self.piezo_thread is None:
            return
        if self.active_display_tab != 0:
            return
        for i in range(3):
            s = self.sensors[i]
            buf = self.piezo_thread.get_buffer_snapshot(s['piezo_adc_group'], s['piezo_channel'])
            if len(buf) == 0:
                continue
            y_data = [v for _, v in buf]
            s['piezo_curve_preview'].setData(y_data)
            n = self.piezo_thread.get_buffer_size(s['piezo_adc_group'], s['piezo_channel'])
            s['piezo_status_label'].setText(f'ADC{s["piezo_adc_group"]+1} CH{s["piezo_channel"]+1} | {n}帧')

    def extract_realtime_piezo_features(self, sensor_idx, current_timestamp=None):
        if self.piezo_thread is None:
            return np.zeros(5, dtype=np.float32)
        s = self.sensors[sensor_idx]
        if current_timestamp is None:
            current_timestamp = time.time()
        window_sec = s['piezo_window_ms'] / 1000.0
        t_start = current_timestamp - window_sec
        window_vals = self.piezo_thread.get_window_values(
            s['piezo_adc_group'], s['piezo_channel'], t_start, current_timestamp)
        if len(window_vals) == 0:
            return np.zeros(5, dtype=np.float32)
        arr = np.array(window_vals, dtype=np.float32)
        return np.array([
            np.mean(arr), np.std(arr),
            np.sqrt(np.mean(arr ** 2)),
            np.max(np.abs(arr)), np.sum(np.abs(arr))
        ], dtype=np.float32)

    # ──────────────────── 六维力传感器 ────────────────────

    def start_ft_subscription(self):
        try:
            ft_options = topic.NodeOptions()
            ft_options.node_name = 'v8_ft_subscriber'
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
            if parm.controller.ftvalues:
                for ftv in parm.controller.ftvalues:
                    raw = [ftv.fx, ftv.fy, ftv.fz, ftv.mx, ftv.my, ftv.mz]
                    self.ft_recent_buffer.append(raw)
                    if self.is_recording:
                        zeroed = [raw[i] - self.ft_bias[i] for i in range(6)]
                        zeroed[2] = -zeroed[2]
                        self.rec_force['timestamps'].append(timestamp)
                        self.rec_force['ft_values'].append(zeroed)

    def _init_force_plot(self, sensor_idx):
        s = self.sensors[sensor_idx]
        init_ylims = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0),
                      (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
        if s['_force_ylims'] is None:
            s['_force_ylims'] = list(init_ylims)
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        x = np.arange(s['force_history_len'])
        axes_force = s['axes_force']
        canvas_force = s['canvas_force']

        s['_force_lines_actual'] = []
        s['_force_lines_pred'] = []
        s['_force_vlines'] = []

        for j, ax in enumerate(axes_force.flat):
            ax.clear()
            line_actual, = ax.plot(x, np.zeros(s['force_history_len']),
                                   'r-', label='实际', linewidth=1.2, alpha=0.7, animated=True)
            line_pred, = ax.plot(x, np.zeros(s['force_history_len']),
                                 'b-', label='预测', linewidth=1.2, alpha=0.7, animated=True)
            vline = ax.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8, animated=True)
            ax.set_title(labels[j], fontsize=9)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, s['force_history_len']])
            ax.set_ylim(init_ylims[j][0], init_ylims[j][1])
            s['_force_lines_actual'].append(line_actual)
            s['_force_lines_pred'].append(line_pred)
            s['_force_vlines'].append(vline)

        s['fig_force'].tight_layout()
        canvas_force.draw()
        s['_force_backgrounds'] = []
        for ax in axes_force.flat:
            s['_force_backgrounds'].append(canvas_force.copy_from_bbox(ax.bbox))
        s['_force_plot_ready'] = True

    def _refresh_force_backgrounds(self, sensor_idx, actual, pred):
        s = self.sensors[sensor_idx]
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for j, ax in enumerate(s['axes_force'].flat):
            ax.set_title(f'{labels[j]}: A={actual[j]:.2f} P={pred[j]:.2f}', fontsize=8)
        s['canvas_force'].draw()
        s['_force_backgrounds'] = []
        for ax in s['axes_force'].flat:
            s['_force_backgrounds'].append(s['canvas_force'].copy_from_bbox(ax.bbox))

    def update_force_plot(self, sensor_idx):
        if self.h5_mode:
            return
        if self.active_display_tab != 0:
            return
        s = self.sensors[sensor_idx]
        if not s['calib_loaded']:
            return

        with self.ft_lock:
            ft_data = self.latest_ft_data
            bias = self.ft_bias.copy()

        actual = np.zeros(6)
        if ft_data and len(ft_data) > 0:
            ftv = ft_data[0]
            actual = np.array([ftv.fx - bias[0], ftv.fy - bias[1], -(ftv.fz - bias[2]),
                              ftv.mx - bias[3], ftv.my - bias[4], ftv.mz - bias[5]])

        pred = s['latest_predicted_force'].copy() if s['latest_predicted_force'] is not None else np.zeros(6)
        pred[2] = -pred[2]

        s['force_actual_history'][s['force_history_idx']] = actual
        s['force_pred_history'][s['force_history_idx']] = pred
        new_vals = np.minimum(actual, pred)
        new_maxs = np.maximum(actual, pred)
        np.minimum(s['_force_data_min'], new_vals, out=s['_force_data_min'])
        np.maximum(s['_force_data_max'], new_maxs, out=s['_force_data_max'])
        s['force_history_idx'] = (s['force_history_idx'] + 1) % s['force_history_len']

        if not s['_force_plot_ready']:
            self._init_force_plot(sensor_idx)
            return

        needs_bg_refresh = False
        for j in range(6):
            data_min = s['_force_data_min'][j]
            data_max = s['_force_data_max'][j]
            lo, hi = s['_force_ylims'][j]
            if data_min < lo or data_max > hi:
                margin = max(abs(data_max - data_min) * 0.3, 0.5)
                init_lo, init_hi = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0),
                                    (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)][j]
                new_lo = min(data_min - margin, init_lo)
                new_hi = max(data_max + margin, init_hi)
                s['_force_ylims'][j] = (new_lo, new_hi)
                s['axes_force'].flat[j].set_ylim(new_lo, new_hi)
                needs_bg_refresh = True

        s['_force_title_counter'] = s.get('_force_title_counter', 0) + 1
        if s['_force_title_counter'] >= 5 or needs_bg_refresh:
            s['_force_title_counter'] = 0
            needs_bg_refresh = True

        if needs_bg_refresh:
            self._refresh_force_backgrounds(sensor_idx, actual, pred)

        show_actual = self.chk_show_actual.isChecked()
        axes_force = s['axes_force']
        canvas_force = s['canvas_force']
        for j, ax in enumerate(axes_force.flat):
            canvas_force.restore_region(s['_force_backgrounds'][j])
            s['_force_lines_actual'][j].set_ydata(s['force_actual_history'][:, j])
            s['_force_lines_pred'][j].set_ydata(s['force_pred_history'][:, j])
            s['_force_vlines'][j].set_xdata([s['force_history_idx'], s['force_history_idx']])
            if show_actual:
                ax.draw_artist(s['_force_lines_actual'][j])
            ax.draw_artist(s['_force_lines_pred'][j])
            ax.draw_artist(s['_force_vlines'][j])
            canvas_force.blit(ax.bbox)

    def update_all_force_plots(self):
        for i in range(3):
            self.update_force_plot(i)

    def on_show_actual_changed(self, state):
        for i in range(3):
            s = self.sensors[i]
            s['_force_plot_ready'] = False

    # ──────────────────── 力传感器调零 ────────────────────

    def auto_zero_ft(self):
        with self.ft_lock:
            buf = list(self.ft_recent_buffer)
        if len(buf) == 0:
            QMessageBox.warning(self, "提示", "尚无力传感器数据，无法调零")
            return
        n = min(50, len(buf))
        recent = np.array(buf[-n:], dtype=np.float64)
        bias = recent.mean(axis=0)
        with self.ft_lock:
            self.ft_bias = bias
        self.status_bar.showMessage(
            f"力传感器已调零: Fx={bias[0]:+.3f} Fy={bias[1]:+.3f} Fz={bias[2]:+.3f} "
            f"Mx={bias[3]:+.3f} My={bias[4]:+.3f} Mz={bias[5]:+.3f}")

    def reset_zero_ft(self):
        with self.ft_lock:
            self.ft_bias = np.zeros(6, dtype=np.float64)
        self.status_bar.showMessage("力传感器调零已重置")



    # ──────────────────── 配置保存/加载 ────────────────────

    def _resolve_path(self, rel_path):
        if rel_path is None:
            return None
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.project_dir, rel_path))

    def _to_relative_path(self, abs_path):
        if abs_path is None:
            return None
        try:
            return os.path.relpath(abs_path, self.project_dir)
        except ValueError:
            return abs_path

    def _update_path_label(self, idx):
        s = self.sensors[idx]
        parts = []
        if s['calib_dir']:
            parts.append(f"标定: {os.path.basename(s['calib_dir'])}")
        if s['data_save_dir']:
            parts.append(f"数据: {os.path.basename(s['data_save_dir'])}")
        if s['model_dir']:
            parts.append(f"模型: {os.path.basename(s['model_dir'])}")
        text = "; ".join(parts) if parts else "未设置"
        self._sensor_path_labels[idx].setText(text)

    def save_input_config(self):
        config = {
            'project_dir': self.project_dir,
            'piezo_port': self.piezo_port,
            'fps': self.spin_fps.value(),
            'match_dist': self.spin_max_dist.value(),
            'sensors': []
        }
        for i in range(3):
            s = self.sensors[i]
            w = self._sensor_widgets[i]
            config['sensors'].append({
                'calib_dir': self._to_relative_path(s['calib_dir']),
                'params_json_path': self._to_relative_path(s['params_json_path']),
                'data_save_dir': self._to_relative_path(s['data_save_dir']),
                'model_dir': self._to_relative_path(s['model_dir']),
                'camera_index': w['spin_cam'].value(),
                'rotate_180': w['chk_rotate'].isChecked(),
                'piezo_adc_group': w['cbb_adc'].currentIndex(),
                'piezo_channel': w['cbb_ch'].currentIndex(),
                'enabled': s['calib_loaded'],
            })
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def load_input_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return 0
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            return 0

        self.project_dir = config.get('project_dir', self.project_dir)
        self.piezo_port = config.get('piezo_port')
        if 'fps' in config:
            self.spin_fps.setValue(config['fps'])
        if 'match_dist' in config:
            self.spin_max_dist.setValue(config['match_dist'])

        loaded_count = 0
        for i, sc in enumerate(config.get('sensors', [])):
            if i >= 3:
                break
            s = self.sensors[i]
            s['calib_dir'] = self._resolve_path(sc.get('calib_dir'))
            s['params_json_path'] = self._resolve_path(sc.get('params_json_path'))
            s['data_save_dir'] = self._resolve_path(sc.get('data_save_dir'))
            s['model_dir'] = self._resolve_path(sc.get('model_dir'))
            s['piezo_adc_group'] = sc.get('piezo_adc_group', i)
            s['piezo_channel'] = sc.get('piezo_channel', i * 2)

            w = self._sensor_widgets[i]
            if 'camera_index' in sc:
                w['spin_cam'].setValue(sc['camera_index'])
            if 'rotate_180' in sc:
                w['chk_rotate'].setChecked(sc['rotate_180'])
            if 'piezo_adc_group' in sc:
                w['cbb_adc'].setCurrentIndex(sc['piezo_adc_group'])
            if 'piezo_channel' in sc:
                w['cbb_ch'].setCurrentIndex(sc['piezo_channel'])

            if sc.get('enabled', False) and s['calib_dir'] and os.path.exists(s['calib_dir']) \
                    and s['data_save_dir'] and os.path.exists(s['data_save_dir']):
                self.load_and_start(i)
                loaded_count += 1
            else:
                self._update_path_label(i)

        return loaded_count

    def auto_load_last_config(self):
        n = self.load_input_config()
        if n == 0:
            self.status_bar.showMessage("就绪 - 请配置各传感器并选择输入源")
        else:
            self.status_bar.showMessage(f"已自动加载 {n} 个传感器配置")

    # ──────────────────── 输入选择 ────────────────────

    def select_input(self, sensor_idx):
        s = self.sensors[sensor_idx]
        # 停止当前传感器线程
        self._stop_sensor_threads(sensor_idx)

        # 1. 标定文件夹
        calib_dir = QFileDialog.getExistingDirectory(
            self, f"Sensor {sensor_idx+1} - 选择标定参数文件夹", "")
        if not calib_dir:
            return
        s['calib_dir'] = calib_dir

        # 2. 参数JSON
        params_path, _ = QFileDialog.getOpenFileName(
            self, f"Sensor {sensor_idx+1} - 选择图像处理参数文件", "",
            "JSON文件 (*.json);;所有文件 (*.*)")
        s['params_json_path'] = params_path if params_path else None

        # 3. data 文件夹
        data_dir = QFileDialog.getExistingDirectory(
            self, f"Sensor {sensor_idx+1} - 选择data文件夹（含roi_masks.npz）", "")
        if not data_dir:
            return
        s['data_save_dir'] = data_dir

        # 4. 力预测模型目录
        default_model_dir = os.path.join(data_dir, "model_output")
        model_dir = QFileDialog.getExistingDirectory(
            self, f"Sensor {sensor_idx+1} - 选择力预测模型目录",
            default_model_dir if os.path.exists(default_model_dir) else data_dir)
        if not model_dir:
            QMessageBox.warning(self, "提示", f"Sensor {sensor_idx+1}: 未选择模型目录，力预测功能将不可用")
        s['model_dir'] = model_dir if model_dir else None

        self._update_path_label(sensor_idx)
        self.save_input_config()
        self.load_and_start(sensor_idx)

    def load_and_start(self, sensor_idx):
        s = self.sensors[sensor_idx]
        try:
            if not self.load_calibration(sensor_idx):
                return
            self.load_detector(sensor_idx)
            if not self.load_reference_data(sensor_idx):
                return
            self.load_force_predictor(sensor_idx)
            self.start_camera(sensor_idx)
            s['enabled'] = True
            self.btn_record.setEnabled(True)
            self.status_bar.showMessage(f"Sensor {sensor_idx+1} 已启动")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1} 启动失败: {e}")
            s['calib_loaded'] = False

    def _stop_sensor_threads(self, sensor_idx):
        s = self.sensors[sensor_idx]
        if s['process_thread'] is not None:
            s['process_thread'].stop()
            s['process_thread'] = None
        if s['camera_thread'] is not None:
            s['camera_thread'].stop()
            s['camera_thread'] = None
        s['calib_loaded'] = False

    # ──────────────────── 力预测模型加载 ────────────────────

    def load_force_predictor(self, sensor_idx):
        s = self.sensors[sensor_idx]
        if not s['model_dir'] or not os.path.exists(s['model_dir']):
            s['force_predictor'] = None
            print(f"Sensor {sensor_idx+1}: 力预测模型未加载")
            return
        try:
            config_path = os.path.join(s['model_dir'], "train_config.json")
            with open(config_path, 'r', encoding='utf-8') as _f:
                _cfg = json.load(_f)
            model_type = _cfg.get('model_type', 'MLP')

            if model_type == 'PointNet':
                from V6_force_predict_pointnet import ForcePredictor
            elif model_type == 'LightNet':
                from V6_force_predict_lightnet import ForcePredictor
            else:
                from V6_force_predict import ForcePredictor

            s['force_predictor'] = ForcePredictor(s['model_dir'])
            print(f"Sensor {sensor_idx+1}: 力预测模型已加载 [{model_type}]: {s['model_dir']}")
        except Exception as e:
            s['force_predictor'] = None
            QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1} 力预测模型加载失败: {e}")

    # ──────────────────── 标定 / 检测器 / 参考数据 ────────────────────

    def load_calibration(self, sensor_idx):
        s = self.sensors[sensor_idx]
        try:
            s['stereo_params'] = self.load_stereo_params(s['calib_dir'])
            s['calib_loaded'] = True
            self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 标定参数已加载")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Sensor {sensor_idx+1} 加载标定参数失败: {e}")
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

    def load_detector(self, sensor_idx):
        s = self.sensors[sensor_idx]
        config_path = s['params_json_path'] if s['params_json_path'] else "marker_params.json"
        s['detector'] = CircleDetector(config_path=config_path, verbose=True)

    def load_reference_data(self, sensor_idx):
        s = self.sensors[sensor_idx]
        roi_file = os.path.join(s['data_save_dir'], "roi_masks.npz")
        if not os.path.exists(roi_file):
            QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1}: 未找到ROI文件: {roi_file}")
            return False
        try:
            with np.load(roi_file) as data:
                left_mask = data['left_mask']
                right_mask = data['right_mask']
                if left_mask.ndim == 3:
                    left_mask = left_mask[:, :, 0]
                if right_mask.ndim == 3:
                    right_mask = right_mask[:, :, 0]
                s['drawing']['left']['mask'] = left_mask
                s['drawing']['right']['mask'] = right_mask
        except Exception as e:
            QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1} ROI加载失败: {e}")
            return False

        mirror_axis = 640
        ref_image_shape = None
        history_file = os.path.join(s['data_save_dir'], "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
                    if 'image_shape' in npz_data:
                        ref_image_shape = tuple(int(v) for v in npz_data['image_shape'])
            except Exception:
                pass

        result_dir = os.path.join(s['data_save_dir'], "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")
        if not os.path.exists(first_frame_file):
            QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1}: 未找到首帧数据: {first_frame_file}")
            return False

        try:
            data = np.genfromtxt(first_frame_file, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 7:
                QMessageBox.warning(self, "警告", f"Sensor {sensor_idx+1}: 首帧数据文件格式错误")
                return False
            points_3d = data[:, :3].astype(np.float64)
            pts1_R = data[:, 3:5].astype(np.float32)
            pts2_R = data[:, 5:7].astype(np.float32)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Sensor {sensor_idx+1} 读取首帧数据失败: {e}")
            return False

        K1, D1 = s['stereo_params']['K1'], s['stereo_params']['D1']
        K2, D2 = s['stereo_params']['K2'], s['stereo_params']['D2']
        if ref_image_shape is not None:
            h, w = ref_image_shape
        else:
            h, w = 480, 1280
        R1, R2, P1, P2 = self._get_stereo_rectify(sensor_idx, w, h)
        pts1 = cv2.undistortPoints(pts1_R, K1, D1, R=R1, P=P1).squeeze()
        pts2 = cv2.undistortPoints(pts2_R, K2, D2, R=R2, P=P2).squeeze()

        fd = s['FRAME_DATA']
        fd.update({
            'initialized': True,
            'roi_masks': {'left': s['drawing']['left']['mask'], 'right': s['drawing']['right']['mask']},
            'P1': P1, 'P2': P2,
            'left_points_0': pts1, 'right_points_0': pts2,
            'left_points_0_R': pts1_R, 'right_points_0_R': pts2_R,
            'left_points_0_pre': pts1_R.copy(), 'right_points_0_pre': pts2_R.copy(),
            'base_3d_points': points_3d,
            'mirror_axis': mirror_axis,
            'ref_image_shape': (h, w),
            'frame_height': h,
        })

        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            fd['transform_origin'] = origin
            fd['transform_rotation'] = rotation_matrix
            s['pca_flip_xy'] = self._detect_pca_orientation(sensor_idx)

        s['drawing']['left']['mask'] = np.zeros((h, w), dtype=np.uint8)
        s['drawing']['right']['mask'] = np.zeros((h, w), dtype=np.uint8)
        left_mask_src = fd['roi_masks']['left']
        right_mask_src = fd['roi_masks']['right']
        if left_mask_src is not None:
            if left_mask_src.shape[:2] != (h, w):
                s['drawing']['left']['mask'] = cv2.resize(left_mask_src.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                s['drawing']['left']['mask'] = left_mask_src.astype(np.uint8)
        if right_mask_src is not None:
            if right_mask_src.shape[:2] != (h, w):
                s['drawing']['right']['mask'] = cv2.resize(right_mask_src.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                s['drawing']['right']['mask'] = right_mask_src.astype(np.uint8)

        fd['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        fd['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        return True


    # ──────────────────── 摄像头启动 ────────────────────

    def start_camera(self, sensor_idx):
        s = self.sensors[sensor_idx]
        w = self._sensor_widgets[sensor_idx]
        s['camera_index'] = w['spin_cam'].value()
        s['rotate_180'] = w['chk_rotate'].isChecked()

        s['process_thread'] = FrameProcessThread()
        s['process_thread'].set_processor(
            lambda frame, idx=sensor_idx: self.process_frame(idx, frame))
        s['process_thread'].set_renderer(
            lambda frame, lp, rp, ll, rl, idx=sensor_idx:
                self._render_to_qimage(idx, frame, lp, rp, ll, rl))
        s['process_thread'].result_signal.connect(
            lambda qi, p3d, lp, rp, ll, rl, ts, abn, _, idx=sensor_idx:
                self.on_process_result(idx, qi, p3d, lp, rp, ll, rl, ts, abn, _))
        s['process_thread'].start()

        s['camera_thread'] = CameraThread(
            camera_index=s['camera_index'],
            fps=self.fps,
            rotate_180=s['rotate_180'])
        s['camera_thread'].frame_signal.connect(
            lambda frame, ts, idx=sensor_idx: self.on_camera_frame(idx, frame, ts))
        s['camera_thread'].fps_signal.connect(
            lambda fps_val, idx=sensor_idx: self.on_camera_fps(idx, fps_val))
        s['camera_thread'].start()

        self.is_playing = True
        s['current_frame_idx'] = 0
        self.status_bar.showMessage(f"Sensor {sensor_idx+1} 摄像头已启动 (ID: {s['camera_index']})")

    def on_camera_fps(self, sensor_idx, fps_val):
        if sensor_idx < len(self._fps_labels):
            self._fps_labels[sensor_idx].setText(f"Cam{sensor_idx+1}: {fps_val:.1f}")

    def on_camera_frame(self, sensor_idx, frame, timestamp=0.0):
        s = self.sensors[sensor_idx]
        if not s['FRAME_DATA']['initialized']:
            return
        s['latest_camera_frame'] = frame
        s['current_frame_idx'] += 1

        h, w = frame.shape[:2]
        ref_shape = s['FRAME_DATA'].get('ref_image_shape')
        if ref_shape is not None and (h, w) != ref_shape:
            if s['current_frame_idx'] == 1:
                QMessageBox.warning(self, "警告",
                    f"Sensor {sensor_idx+1}: 摄像头分辨率 {w}x{h} 与首帧数据分辨率不一致")

        if s['current_frame_idx'] % 10 == 0 and sensor_idx == 0:
            self.lbl_frame_info.setText(f"帧: {s['current_frame_idx']}")

        if self.is_playing and s['process_thread'] is not None:
            s['process_thread'].add_frame(frame.copy(), timestamp)
        else:
            self.display_frame(sensor_idx, frame)

    def on_process_result(self, sensor_idx, qt_image, points_3d, _lp, _rp, _ll, _rl,
                          timestamp=0.0, is_abnormal=False, _=None):
        if points_3d is None:
            return

        s = self.sensors[sensor_idx]
        predicted_force = self._predict_force(sensor_idx, points_3d)
        s['latest_predicted_force'] = predicted_force

        self.buffer_frame_data(sensor_idx, points_3d, timestamp, is_abnormal, predicted_force)

        # 视频渲染：仅在"视频画面"页激活时
        if self.active_display_tab == 1 and qt_image is not None:
            self._set_video_pixmap(sensor_idx, qt_image)

        # 点云+力曲线+压电更新：仅在"传感器数据"页激活时
        if self.active_display_tab != 0:
            return

        s['pointcloud_frame_counter'] += 1
        if s['pointcloud_frame_counter'] < s['pointcloud_update_interval']:
            return
        s['pointcloud_frame_counter'] = 0

        if s['is_dragging']:
            return
        if time.time() - s['drag_release_time'] < 0.5:
            return
        current_time = time.time()
        if current_time - s['last_3d_update_time'] >= s['min_3d_update_interval']:
            self.update_3d_view(sensor_idx, points_3d)
            s['last_3d_update_time'] = current_time

    # ──────────────────── 力预测 ────────────────────

    def _predict_force(self, sensor_idx, points_3d):
        s = self.sensors[sensor_idx]
        if s['force_predictor'] is None:
            return None
        base_3d = s['FRAME_DATA']['base_3d_points']
        if base_3d is None or len(points_3d) != len(base_3d):
            return None
        try:
            local_xyz = self.get_local_coords(sensor_idx, points_3d)
            base_local = self.get_local_coords(sensor_idx, base_3d)
            dxyz = (local_xyz - base_local).astype(np.float32)

            piezo_feat = None
            if s['force_predictor'].config.get('use_piezo', False):
                piezo_feat = self.extract_realtime_piezo_features(sensor_idx)

            return s['force_predictor'].predict(dxyz, piezo_feat)
        except Exception as e:
            print(f"Sensor {sensor_idx+1} 力预测失败: {e}")
            return None

    # ──────────────────── 录制 ────────────────────

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        any_initialized = any(s['FRAME_DATA']['initialized'] for s in self.sensors)
        if not any_initialized:
            self.status_bar.showMessage("请先初始化传感器")
            return
        self.rec_vision = [
            {'timestamps': [], 'xyz': [], 'abnormal': [], 'predicted_force': []}
            for _ in range(3)
        ]
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

    def get_local_coords(self, sensor_idx, points_3d):
        s = self.sensors[sensor_idx]
        if s['FRAME_DATA']['transform_rotation'] is not None:
            return self.transform_to_local_coordinates(sensor_idx, points_3d)
        return points_3d.copy()

    def buffer_frame_data(self, sensor_idx, points_3d, timestamp, is_abnormal=False, predicted_force=None):
        if not self.is_recording or points_3d is None:
            return
        s = self.sensors[sensor_idx]
        local_xyz = self.get_local_coords(sensor_idx, points_3d)
        buf = self.rec_vision[sensor_idx]
        buf['timestamps'].append(timestamp)
        buf['xyz'].append(local_xyz.astype(np.float32))
        buf['abnormal'].append(is_abnormal)
        if predicted_force is not None:
            pf = predicted_force.astype(np.float32).copy()
            pf[2] = -pf[2]
            buf['predicted_force'].append(pf)
        else:
            out_dim = s['force_predictor'].config['output_dim'] if s['force_predictor'] else 6
            buf['predicted_force'].append(np.zeros(out_dim, dtype=np.float32))

    def save_unified_hdf5(self):
        stop_time = datetime.now()

        all_vis_ts = []
        for i in range(3):
            all_vis_ts.extend(self.rec_vision[i]['timestamps'])
        with self.ft_lock:
            ft_ts = self.rec_force['timestamps'].copy()
            ft_vals = self.rec_force['ft_values'].copy()

        if not all_vis_ts and not ft_ts:
            QMessageBox.warning(self, "警告", "没有采集到任何数据")
            self.status_bar.showMessage("采集已停止（无数据）")
            return

        filename = f"V8_recording_{stop_time.strftime('%Y%m%d_%H%M%S')}.h5"
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        try:
            with h5py.File(filepath, 'w') as f:
                for sensor_idx in range(3):
                    s = self.sensors[sensor_idx]
                    buf = self.rec_vision[sensor_idx]
                    vis_ts = buf['timestamps']
                    if not vis_ts:
                        continue

                    grp = f.create_group(f'sensor_{sensor_idx}')
                    timestamps_v = np.array(vis_ts, dtype=np.float64)
                    xyz_all = np.array(buf['xyz'], dtype=np.float32)
                    abnormal = np.array(buf['abnormal'], dtype=np.bool_)
                    pred_force = np.array(buf['predicted_force'], dtype=np.float32)

                    base_3d = s['FRAME_DATA']['base_3d_points']
                    xyz_ref = self.get_local_coords(sensor_idx, base_3d).astype(np.float32)
                    dxyz = (xyz_all - xyz_ref[np.newaxis, :, :]).astype(np.float32)
                    N = xyz_ref.shape[0]

                    vision = grp.create_group('vision')
                    vision.create_dataset('timestamp', data=timestamps_v)
                    vision.create_dataset('xyz', data=xyz_all)
                    vision.create_dataset('dxyz', data=dxyz)
                    vision.create_dataset('abnormal', data=abnormal)
                    vision.create_dataset('predicted_force', data=pred_force)
                    vision.create_dataset('point_id', data=np.arange(N, dtype=np.int32))

                    ref = grp.create_group('reference')
                    ref.create_dataset('xyz_ref', data=xyz_ref)

                if ft_ts:
                    force = f.create_group('force')
                    force.create_dataset('timestamp', data=np.array(ft_ts, dtype=np.float64))
                    force.create_dataset('values', data=np.array(ft_vals, dtype=np.float64))
                    force.attrs['columns'] = 'fx,fy,fz,mx,my,mz'

                meta = f.create_group('meta')
                meta.attrs['camera_fps'] = self.fps
                meta.attrs['experiment_name'] = filename
                meta.attrs['ft_bias'] = self.ft_bias.tolist()
                for i in range(3):
                    if self.sensors[i]['calib_loaded']:
                        n_pts = len(self.rec_vision[i]['xyz'][0]) if self.rec_vision[i]['xyz'] else 0
                        meta.attrs[f'sensor_{i}_marker_count'] = n_pts
                        meta.attrs[f'sensor_{i}_vision_frames'] = len(self.rec_vision[i]['timestamps'])
                meta.attrs['force_samples'] = len(ft_ts)

            self.rec_vision = [
                {'timestamps': [], 'xyz': [], 'abnormal': [], 'predicted_force': []}
                for _ in range(3)
            ]
            with self.ft_lock:
                self.rec_force = {'timestamps': [], 'ft_values': []}

            QMessageBox.information(self, "保存成功", f"数据已保存:\\n{filepath}")
            self.status_bar.showMessage(f"采集完成: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入HDF5失败:\\n{e}")
            self.status_bar.showMessage(f"保存失败: {e}")



    # ──────────────────── 帧处理 (检测+重建) ────────────────────

    def process_frame(self, sensor_idx, frame):
        s = self.sensors[sensor_idx]
        fd = s['FRAME_DATA']
        if not fd['initialized']:
            return None, None, None, None, None, False

        h, w = frame.shape[:2]
        mirror_axis = fd['mirror_axis']

        if s['_left_img_buf'] is None or s['_left_img_buf'].shape != frame.shape:
            s['_left_img_buf'] = np.full((h, w, 3), 255, dtype=np.uint8)
            s['_right_img_buf'] = np.full((h, w, 3), 255, dtype=np.uint8)
        s['_left_img_buf'][:] = 255
        s['_left_img_buf'][:, :mirror_axis] = frame[:, :mirror_axis]
        s['_right_img_buf'][:] = 255
        s['_right_img_buf'][:, mirror_axis:] = frame[:, mirror_axis:]
        left_img = s['_left_img_buf']
        right_img = s['_right_img_buf']

        left_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                     self.apply_roi_mask(s['detector'].detect(left_img),
                                                         s['drawing']['left']['mask'])], dtype=np.float32)
        right_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                      self.apply_roi_mask(s['detector'].detect(right_img),
                                                          s['drawing']['right']['mask'])], dtype=np.float32)

        pre_l = fd['left_points_0_pre']
        pre_r = fd['right_points_0_pre']
        filter_dist = self.max_match_dist * 1.5

        if id(pre_l) != s['_pre_l_tree_id']:
            s['_pre_l_tree'] = cKDTree(pre_l) if len(pre_l) > 0 else None
            s['_pre_l_tree_id'] = id(pre_l)
        if id(pre_r) != s['_pre_r_tree_id']:
            s['_pre_r_tree'] = cKDTree(pre_r) if len(pre_r) > 0 else None
            s['_pre_r_tree_id'] = id(pre_r)

        if len(left_pts_det_raw) > 0 and s['_pre_l_tree'] is not None:
            min_dists_l, _ = s['_pre_l_tree'].query(left_pts_det_raw, k=1)
            left_pts_det = left_pts_det_raw[min_dists_l <= filter_dist]
        else:
            left_pts_det = left_pts_det_raw

        if len(right_pts_det_raw) > 0 and s['_pre_r_tree'] is not None:
            min_dists_r, _ = s['_pre_r_tree'].query(right_pts_det_raw, k=1)
            right_pts_det = right_pts_det_raw[min_dists_r <= filter_dist]
        else:
            right_pts_det = right_pts_det_raw

        matched_pairs = self.auto_match_points(
            left_pts_det, right_pts_det,
            fd['left_points_0_pre'], fd['right_points_0_pre'],
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

        left_edges = fd.get('left_edges') or []
        right_edges = fd.get('right_edges') or []

        def find_crossing_pairs(pts, edges):
            crossings = []
            n = len(edges)
            if n == 0:
                return crossings
            bboxes = np.empty((n, 4), dtype=np.float64)
            for i, (a, b) in enumerate(edges):
                ax, ay = pts[a][0], pts[a][1]
                bx, by = pts[b][0], pts[b][1]
                bboxes[i, 0] = min(ax, bx); bboxes[i, 1] = max(ax, bx)
                bboxes[i, 2] = min(ay, by); bboxes[i, 3] = max(ay, by)
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
                    a, b = edges[ei]; c, d = edges[ej]
                    candidates = [(a, c), (a, d), (b, c), (b, d)]
                    for p, q in candidates:
                        p_lost = lost_mask[p]; q_lost = lost_mask[q]
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
                                base_pts = fd['left_points_0_R'] if side == 'left' else fd['right_points_0_R']
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

        num_total = len(pre_l)
        num_lost = int(np.sum(left_lost_mask) + np.sum(right_lost_mask))
        lost_ratio = num_lost / (num_total * 2) if num_total > 0 else 1.0
        left_initial_crossings = find_crossing_pairs(left_points_R, left_edges) if left_edges else []
        right_initial_crossings = find_crossing_pairs(right_points_R, right_edges) if right_edges else []
        is_catastrophic = (lost_ratio > 0.5 or
                           len(left_initial_crossings) > 20 or
                           len(right_initial_crossings) > 20)

        if is_catastrophic:
            fd['current_left_crossings'] = left_initial_crossings
            fd['current_right_crossings'] = right_initial_crossings
        else:
            if left_edges:
                untangle(left_points_R, left_lost_mask, left_edges, matched_pairs, 'left')
            if right_edges:
                untangle(right_points_R, right_lost_mask, right_edges, matched_pairs, 'right')
            fd['current_left_crossings'] = find_crossing_pairs(left_points_R, left_edges) if left_edges else []
            fd['current_right_crossings'] = find_crossing_pairs(right_points_R, right_edges) if right_edges else []

        if not is_catastrophic:
            left_points_R = left_points_R.astype(np.float32)
            right_points_R = right_points_R.astype(np.float32)
            K1, D1 = s['stereo_params']['K1'], s['stereo_params']['D1']
            K2, D2 = s['stereo_params']['K2'], s['stereo_params']['D2']
            R1, R2, P1, P2 = self._get_stereo_rectify(sensor_idx, w, h)
            l_pts_ud = cv2.undistortPoints(left_points_R, K1, D1, R=R1, P=P1).squeeze()
            r_pts_ud = cv2.undistortPoints(right_points_R, K2, D2, R=R2, P=P2).squeeze()
            points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

            base_3d = fd['base_3d_points']
            is_abnormal = False
            if base_3d is not None and len(points_3d) == len(base_3d):
                z_diff = points_3d[:, 2] - base_3d[:, 2]
                if np.any(z_diff > 1.0):
                    is_abnormal = True
                if not is_abnormal and len(points_3d) >= 2:
                    pairs = cKDTree(points_3d).query_pairs(0.3)
                    if len(pairs) > 0:
                        is_abnormal = True
        else:
            is_abnormal = True
            base_3d = fd['base_3d_points']

        if not is_abnormal:
            s['consecutive_abnormal_frames'] = 0
            fd['left_points_0_pre'] = left_points_R
            fd['right_points_0_pre'] = right_points_R
            return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask, False

        s['consecutive_abnormal_frames'] += 1

        if s['consecutive_abnormal_frames'] >= 5:
            fd['left_points_0_pre'] = fd['left_points_0_R'].copy()
            fd['right_points_0_pre'] = fd['right_points_0_R'].copy()
            s['consecutive_abnormal_frames'] = 0
            n_pts = len(base_3d) if base_3d is not None else len(fd['left_points_0_R'])
            return (fd['base_3d_points'].copy(),
                    fd['left_points_0_R'].copy(), fd['right_points_0_R'].copy(),
                    np.ones(n_pts, dtype=bool), np.ones(n_pts, dtype=bool), True)
        else:
            prev_left = fd['left_points_0_pre']
            prev_right = fd['right_points_0_pre']
            K1, D1 = s['stereo_params']['K1'], s['stereo_params']['D1']
            K2, D2 = s['stereo_params']['K2'], s['stereo_params']['D2']
            R1, R2, P1_prev, P2_prev = self._get_stereo_rectify(sensor_idx, w, h)
            l_pts_ud_prev = cv2.undistortPoints(prev_left, K1, D1, R=R1, P=P1_prev).squeeze()
            r_pts_ud_prev = cv2.undistortPoints(prev_right, K2, D2, R=R2, P=P2_prev).squeeze()
            points_3d_prev = self.linear_triangulation(l_pts_ud_prev, r_pts_ud_prev, P1_prev, P2_prev)
            n_pts = len(base_3d) if base_3d is not None else len(prev_left)
            return (points_3d_prev, prev_left.copy(), prev_right.copy(),
                    np.ones(n_pts, dtype=bool), np.ones(n_pts, dtype=bool), True)

    # ──────────────────── 显示 ────────────────────

    def _render_to_qimage(self, sensor_idx, frame, left_pts=None, right_pts=None,
                          left_lost=None, right_lost=None):
        s = self.sensors[sensor_idx]
        display_img = frame.copy()

        left_edges = s['FRAME_DATA'].get('left_edges') or []
        right_edges = s['FRAME_DATA'].get('right_edges') or []
        if left_pts is not None and left_edges:
            crossing_left = set()
            for ei, ej in s['FRAME_DATA'].get('current_left_crossings', []):
                crossing_left.add(ei); crossing_left.add(ej)
            for k, (a, b) in enumerate(left_edges):
                pa = (int(left_pts[a][0]), int(left_pts[a][1]))
                pb = (int(left_pts[b][0]), int(left_pts[b][1]))
                color = (0, 0, 220) if k in crossing_left else (200, 180, 80)
                cv2.line(display_img, pa, pb, color, 1, cv2.LINE_AA)
        if right_pts is not None and right_edges:
            crossing_right = set()
            for ei, ej in s['FRAME_DATA'].get('current_right_crossings', []):
                crossing_right.add(ei); crossing_right.add(ej)
            for k, (a, b) in enumerate(right_edges):
                pa = (int(right_pts[a][0]), int(right_pts[a][1]))
                pb = (int(right_pts[b][0]), int(right_pts[b][1]))
                color = (0, 0, 220) if k in crossing_right else (80, 200, 80)
                cv2.line(display_img, pa, pb, color, 1, cv2.LINE_AA)

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
        return QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def display_frame(self, sensor_idx, frame, left_pts=None, right_pts=None,
                      left_lost=None, right_lost=None):
        qt_image = self._render_to_qimage(sensor_idx, frame, left_pts, right_pts, left_lost, right_lost)
        self._set_video_pixmap(sensor_idx, qt_image)

    def _set_video_pixmap(self, sensor_idx, qt_image):
        if self.active_display_tab != 1:
            return
        lbl = self.sensors[sensor_idx].get('lbl_video')
        if lbl is None:
            return
        label_size = lbl.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        lbl.setPixmap(scaled_pixmap)

    def update_3d_view(self, sensor_idx, points_3d):
        if self.active_display_tab != 0:
            return
        s = self.sensors[sensor_idx]
        if s['is_dragging']:
            return
        if time.time() - s['drag_release_time'] < 0.5:
            return

        ax = s['ax_3d']
        fig_3d = s['fig_3d']

        if s['FRAME_DATA']['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(sensor_idx, points_3d)
            base_local = self.transform_to_local_coordinates(sensor_idx, s['FRAME_DATA']['base_3d_points'])
            displacement_vectors = points - base_local
            deformation = np.linalg.norm(displacement_vectors, axis=1)

            if s['_scatter_plot'] is None:
                fig_3d.clear()
                ax = fig_3d.add_subplot(111, projection='3d')
                s['ax_3d'] = ax
                s['_scatter_plot'] = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                c=deformation, cmap='jet', s=30, vmin=0, vmax=1.5)
                s['_colorbar'] = fig_3d.colorbar(s['_scatter_plot'], ax=ax, shrink=0.8)
                s['_colorbar'].set_label('Deformation (mm)', rotation=270, labelpad=15)
            else:
                s['_scatter_plot']._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                s['_scatter_plot'].set_array(deformation)

            if s['_force_arrow_plot'] is not None:
                try:
                    s['_force_arrow_plot'].remove()
                except (ValueError, AttributeError):
                    pass
                s['_force_arrow_plot'] = None

            if s['latest_predicted_force'] is not None:
                pred_force = s['latest_predicted_force'][:3]
                force_mag = np.linalg.norm(pred_force)
                if force_mag > 0.5:
                    center = np.mean(points, axis=0)
                    force_dir = pred_force / force_mag
                    arrow_len = force_mag * 2.0
                    s['_force_arrow_plot'] = ax.quiver(
                        center[0], center[1], center[2],
                        force_dir[0] * arrow_len, force_dir[1] * arrow_len, force_dir[2] * arrow_len,
                        color='blue', arrow_length_ratio=0.2, linewidth=2,
                        label=f'F={force_mag:.1f}N')
                    ax.legend(loc='upper left', fontsize=7)
        else:
            points = points_3d.copy()
            points[:, 0] = -points[:, 0]
            points[:, 1] = -points[:, 1]
            if s['_scatter_plot'] is None:
                fig_3d.clear()
                ax = fig_3d.add_subplot(111, projection='3d')
                s['ax_3d'] = ax
                s['_scatter_plot'] = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=30)
            else:
                s['_scatter_plot']._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'S{sensor_idx+1} Frame {s["current_frame_idx"]}', fontsize=9)

        if len(points) > 0:
            ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])

        if s['view_saved']:
            try:
                ax.view_init(elev=s['view_elev'], azim=s['view_azim'], roll=s['view_roll'])
            except TypeError:
                ax.view_init(elev=s['view_elev'], azim=s['view_azim'])
        else:
            try:
                ax.view_init(elev=90, azim=180, roll=-90)
            except TypeError:
                ax.view_init(elev=90, azim=180)

        s['canvas_3d'].draw_idle()


    # ──────────────────── 工具函数 ────────────────────

    def _get_stereo_rectify(self, sensor_idx, w, h):
        s = self.sensors[sensor_idx]
        key = (w, h)
        if key not in s['_stereo_rectify_cache']:
            K1, D1 = s['stereo_params']['K1'], s['stereo_params']['D1']
            K2, D2 = s['stereo_params']['K2'], s['stereo_params']['D2']
            R, T = s['stereo_params']['R'], s['stereo_params']['T']
            R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
                K1, D1, K2, D2, (w, h), R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            s['_stereo_rectify_cache'][key] = (R1, R2, P1, P2)
        return s['_stereo_rectify_cache'][key]

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
        pts1_h = pts1.T.reshape(2, -1).astype(np.float64)
        pts2_h = pts2.T.reshape(2, -1).astype(np.float64)
        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        pts4d /= pts4d[3]
        return pts4d[:3].T

    @staticmethod
    def extract_edges(points_2d):
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
            pre_dists, _ = cKDTree(ref_pts).query(det_pts, k=1)
            det_pts = det_pts[pre_dists <= threshold]
            if len(det_pts) == 0:
                return {}
            dist_matrix = np.linalg.norm(ref_pts[:, None] - det_pts[None], axis=2)
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

    def build_coordinate_system_pca(self, points_3d, prev_rotation=None):
        centroid = np.mean(points_3d, axis=0)
        centered = points_3d - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        x_axis = eigenvectors[:, 0]
        y_axis = eigenvectors[:, 1]
        z_axis = eigenvectors[:, 2]

        if prev_rotation is not None:
            prev_x = prev_rotation[:, 0]
            prev_z = prev_rotation[:, 2]
            if np.dot(z_axis, prev_z) < 0:
                z_axis = -z_axis
            if np.dot(x_axis, prev_x) < 0:
                x_axis = -x_axis
        else:
            if z_axis[2] < 0:
                z_axis = -z_axis
            if x_axis[0] < 0:
                x_axis = -x_axis

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        return centroid, rotation_matrix

    def _detect_pca_orientation(self, sensor_idx):
        s = self.sensors[sensor_idx]
        fd = s['FRAME_DATA']
        origin = fd.get('transform_origin')
        rotation = fd.get('transform_rotation')
        points_3d = fd.get('base_3d_points')
        if origin is None or rotation is None or points_3d is None:
            return False
        local = np.dot(points_3d - origin, rotation)
        return np.sum(local[:, 1] > 6) == 0

    def transform_to_local_coordinates(self, sensor_idx, points):
        s = self.sensors[sensor_idx]
        fd = s['FRAME_DATA']
        origin = fd['transform_origin']
        rotation = fd['transform_rotation']
        translated = points - origin
        local = np.dot(translated, rotation)
        if s['pca_flip_xy']:
            local[:, 0] = -local[:, 0]
            local[:, 1] = -local[:, 1]
        local[:, 0] = -local[:, 0]
        return local

    # ──────────────────── 3D 视图鼠标事件 ────────────────────

    def on_3d_mouse_press(self, sensor_idx, event):
        s = self.sensors[sensor_idx]
        s['is_dragging'] = True
        s['_was_playing_before_drag'] = self.is_playing

    def on_3d_mouse_motion(self, sensor_idx, event):
        s = self.sensors[sensor_idx]
        if s['is_dragging']:
            ax = s['ax_3d']
            s['view_elev'] = ax.elev
            s['view_azim'] = ax.azim
            s['view_roll'] = getattr(ax, 'roll', 0)
            s['view_saved'] = True

    def on_3d_mouse_release(self, sensor_idx, event):
        s = self.sensors[sensor_idx]
        if s['is_dragging']:
            s['is_dragging'] = False
            s['drag_release_time'] = time.time()
            ax = s['ax_3d']
            s['view_elev'] = ax.elev
            s['view_azim'] = ax.azim
            s['view_roll'] = getattr(ax, 'roll', 0)
            s['view_saved'] = True
            if s['_was_playing_before_drag']:
                s['_was_playing_before_drag'] = False

    # ──────────────────── 播放速度 ────────────────────

    def update_playback_speed(self, fps_val):
        self.fps = fps_val
        for i in range(3):
            s = self.sensors[i]
            if s['camera_thread'] is not None:
                s['camera_thread'].target_fps = fps_val

    # ──────────────────── 重置首帧 / 设为首帧 ────────────────────

    def reset_first_frame(self, sensor_idx):
        s = self.sensors[sensor_idx]
        fd = s['FRAME_DATA']
        if not fd['initialized']:
            self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 请先初始化首帧")
            return
        s['current_frame_idx'] = 0
        fd['left_points_0_pre'] = fd['left_points_0_R'].copy()
        fd['right_points_0_pre'] = fd['right_points_0_R'].copy()
        s['consecutive_abnormal_frames'] = 0
        s['_scatter_plot'] = None
        s['_colorbar'] = None
        s['_force_arrow_plot'] = None
        s['fig_3d'].clear()
        s['ax_3d'] = s['fig_3d'].add_subplot(111, projection='3d')
        s['canvas_3d'].draw_idle()
        self.update_3d_view(sensor_idx, fd['base_3d_points'])
        self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 已重置参考点")

    def set_current_as_base(self, sensor_idx):
        s = self.sensors[sensor_idx]
        fd = s['FRAME_DATA']
        if not fd['initialized']:
            self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 请先初始化首帧")
            return
        left_pts = fd.get('left_points_0_pre')
        right_pts = fd.get('right_points_0_pre')
        if left_pts is None or right_pts is None:
            self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 当前帧数据不可用")
            return

        fd['left_points_0_R'] = left_pts.copy()
        fd['right_points_0_R'] = right_pts.copy()

        K1, D1 = s['stereo_params']['K1'], s['stereo_params']['D1']
        K2, D2 = s['stereo_params']['K2'], s['stereo_params']['D2']
        w = fd['mirror_axis'] * 2
        h = fd.get('frame_height', 480)
        R1, R2, P1, P2 = self._get_stereo_rectify(sensor_idx, w, h)
        l_pts_ud = cv2.undistortPoints(left_pts, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_pts, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)
        fd['base_3d_points'] = points_3d

        if len(points_3d) >= 3:
            prev_rot = fd.get('transform_rotation')
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d, prev_rotation=prev_rot)
            fd['transform_origin'] = origin
            fd['transform_rotation'] = rotation_matrix

        fd['left_edges'] = self.extract_edges(left_pts.astype(np.float64))
        fd['right_edges'] = self.extract_edges(right_pts.astype(np.float64))

        s['_scatter_plot'] = None
        s['_colorbar'] = None
        if s['_force_arrow_plot'] is not None:
            try:
                s['_force_arrow_plot'].remove()
            except (ValueError, AttributeError):
                pass
            s['_force_arrow_plot'] = None
        s['fig_3d'].clear()
        s['ax_3d'] = s['fig_3d'].add_subplot(111, projection='3d')
        s['canvas_3d'].draw_idle()

        s['consecutive_abnormal_frames'] = 0
        s['force_actual_history'] = np.zeros((s['force_history_len'], 6))
        s['force_pred_history'] = np.zeros((s['force_history_len'], 6))
        s['force_history_idx'] = 0
        s['_force_data_min'] = np.zeros(6)
        s['_force_data_max'] = np.zeros(6)
        s['_force_ylims'] = None
        s['_force_plot_ready'] = False
        s['_force_title_counter'] = 0

        self.status_bar.showMessage(f"Sensor {sensor_idx+1}: 已将当前帧设置为新的基准帧")

    # ──────────────────── HDF5回放 ────────────────────

    def load_h5_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择HDF5文件', '', 'HDF5 Files (*.h5 *.hdf5)')
        if not file_path:
            return
        try:
            if self.h5_file:
                self.h5_file.close()
            self.h5_file = h5py.File(file_path, 'r')
            self.h5_mode = True
            self.status_bar.showMessage(f'已加载HDF5: {os.path.basename(file_path)}')
            QMessageBox.information(self, "提示",
                "HDF5回放为简化模式。\\n请使用 V7_visualize_compare.py 进行完整回放分析。")
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载HDF5失败：{str(e)}')

    # ──────────────────── 关闭 ────────────────────

    def closeEvent(self, event):
        if self.h5_file:
            self.h5_file.close()
        if self.is_recording:
            self.stop_recording()
        self.force_plot_timer.stop()
        self._piezo_timer.stop()

        if self.piezo_thread is not None:
            self.piezo_thread.stop()
            self.piezo_thread = None

        if self.ft_node is not None:
            self.ft_subscription = None
            try:
                self.ft_node.Shutdown()
            except Exception:
                pass
            self.ft_node = None

        for i in range(3):
            s = self.sensors[i]
            if s['process_thread'] is not None:
                s['process_thread'].running = False
                s['process_thread'].has_new_frame.set()
            if s['camera_thread'] is not None:
                s['camera_thread'].running = False
            if s['process_thread'] is not None:
                s['process_thread'].stop()
                s['process_thread'] = None
            if s['camera_thread'] is not None:
                s['camera_thread'].stop()
                s['camera_thread'] = None

        event.accept()


# ─────────────────────── main ───────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    window = V8MainWindow()
    window.show()
    ret = app.exec_()
    os._exit(ret)
