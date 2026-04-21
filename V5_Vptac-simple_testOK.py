# -*- coding: utf-8 -*-
'''
视频/摄像头标记点跟踪与三维重建播放器（简化版）
功能说明：
1. 支持读取视频文件或USB摄像头实时采集
2. 逐帧进行标记点检测和三维重建
3. 启动前弹出文件选择窗口：标定文件夹、参数JSON文件、首帧ROI和匹配数据保存文件夹
4. GUI中同时显示视频帧和三维点云
5. 支持暂停/播放，支持手动调整三维点云观察角度
6. 支持"设置为首帧"动态更新基准帧
'''
import sys
import signal
import os
import json
import time
import threading
from datetime import datetime
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QGroupBox, QStatusBar, QSlider, QSpinBox, QSplitter,
                             QCheckBox)
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


class CameraThread(QThread):
    """摄像头采集线程"""
    frame_signal = pyqtSignal(np.ndarray, float)

    def __init__(self, camera_index=0, fps=60, rotate_180=True):
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


class FrameProcessThread(QThread):
    """帧处理线程"""
    result_signal = pyqtSignal(object, object, object, object, object, object, object)

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
                    points_3d, left_pts, right_pts, left_lost, right_lost = self.process_func(frame)
                    self.result_signal.emit(frame, points_3d, left_pts, right_pts, left_lost, right_lost, timestamp)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


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
    """视频/摄像头点云播放器主窗口（简化版）"""
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_input_config.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉系统 - 标记点跟踪与三维重建（简化版）")
        self.setGeometry(50, 50, 1600, 900)

        self.input_mode = None
        self.video_path = None
        self.calib_dir = None
        self.params_json_path = None
        self.data_save_dir = None

        self.video_cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 60

        self.camera_thread = None
        self.camera_index = 0
        self.rotate_180 = True
        self.latest_camera_frame = None

        self.process_thread = None
        self.last_3d_update_time = 0
        self.min_3d_update_interval = 0.1

        self.stereo_params = None
        self.calib_loaded = False

        self.FRAME_DATA = {
            'initialized': False, 'roi_masks': None, 'P1': None, 'P2': None,
            'left_points_0': None, 'right_points_0': None,
            'left_points_0_R': None, 'right_points_0_R': None,
            'left_points_0_pre': None, 'right_points_0_pre': None,
            'base_3d_points': None, 'mirror_axis': None,
            'transform_origin': None, 'transform_rotation': None,
            'left_edges': None, 'right_edges': None,
            'current_left_crossings': [], 'current_right_crossings': [],
        }
        self.drawing = {'left': {'mask': None}, 'right': {'mask': None}}

        self.detector = None
        self.max_match_dist = 50

        self.view_elev = 90
        self.view_azim = 0
        self.view_roll = -90
        self.view_saved = False
        self.is_dragging = False
        self.drag_release_time = 0
        self.drag_cooldown = 0.5
        self._was_playing_before_drag = False

        self.is_playing = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.on_timer_tick)

        self.avg_displacement_vector = np.array([0.0, 0.0, 0.0])
        self.avg_displacement_magnitude = 0.0
        self.max_displacement = 0.0
        self.min_displacement = 0.0
        self.current_points_3d = None
        self.consecutive_abnormal_frames = 0  # 连续异常帧计数器（灾难性熔断与自动重置使用）

        self.init_ui()
        QTimer.singleShot(100, self.auto_load_last_config)

    def init_ui(self):
        """初始化界面"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)

        self.btn_select_input = QPushButton("输入源")
        self.btn_select_input.clicked.connect(self.select_input_mode)

        self.btn_play_pause = QPushButton("播放")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_first_frame)
        self.btn_reset.setEnabled(False)

        self.btn_set_as_base = QPushButton("设为基准")
        self.btn_set_as_base.clicked.connect(self.set_current_as_base)
        self.btn_set_as_base.setEnabled(False)

        self.lbl_frame_info = QLabel("帧: 0")
        self.lbl_frame_info.setMinimumWidth(80)

        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 120)
        self.spin_speed.setValue(60)
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.setMaximumWidth(80)
        self.spin_speed.valueChanged.connect(self.update_playback_speed)

        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 500)
        self.spin_max_dist.setValue(50)
        self.spin_max_dist.setPrefix("匹配:")
        self.spin_max_dist.setMaximumWidth(90)
        self.spin_max_dist.valueChanged.connect(self.update_max_dist)

        self.lbl_camera = QLabel("摄像头:")
        self.spin_camera_idx = QSpinBox()
        self.spin_camera_idx.setRange(0, 10)
        self.spin_camera_idx.setValue(0)
        self.spin_camera_idx.setMaximumWidth(50)

        self.chk_rotate = QCheckBox("旋转180°")
        self.chk_rotate.setChecked(True)

        self.chk_show_avg_disp = QCheckBox("显示平均位移")
        self.chk_show_avg_disp.setChecked(True)

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
        control_layout.addWidget(self.chk_show_avg_disp)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        self.slider_progress = QSlider(Qt.Horizontal)
        self.slider_progress.setRange(0, 100)
        self.slider_progress.setValue(0)
        self.slider_progress.sliderPressed.connect(self.on_slider_pressed)
        self.slider_progress.sliderReleased.connect(self.on_slider_released)
        self.slider_progress.valueChanged.connect(self.on_slider_changed)
        self.slider_dragging = False
        main_layout.addWidget(self.slider_progress)

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
        main_layout.addWidget(visual_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请选择视频和配置文件")

        self.canvas_3d.mpl_connect('button_press_event', self.on_3d_mouse_press)
        self.canvas_3d.mpl_connect('button_release_event', self.on_3d_mouse_release)
        self.canvas_3d.mpl_connect('motion_notify_event', self.on_3d_mouse_motion)

    def save_input_config(self):
        config = {'input_mode': self.input_mode, 'video_path': self.video_path,
                  'calib_dir': self.calib_dir, 'params_json_path': self.params_json_path,
                  'data_save_dir': self.data_save_dir, 'camera_index': self.spin_camera_idx.value(),
                  'rotate_180': self.chk_rotate.isChecked()}
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
            if config.get('input_mode') not in ['video', 'camera']:
                return False
            if not config.get('calib_dir') or not os.path.exists(config['calib_dir']):
                return False
            if not config.get('data_save_dir') or not os.path.exists(config['data_save_dir']):
                return False
            if config['input_mode'] == 'video':
                if not config.get('video_path') or not os.path.exists(config['video_path']):
                    return False
                self.video_path = config['video_path']
            self.input_mode = config['input_mode']
            self.calib_dir = config['calib_dir']
            self.data_save_dir = config['data_save_dir']
            self.params_json_path = config.get('params_json_path')
            if 'camera_index' in config:
                self.spin_camera_idx.setValue(config['camera_index'])
            if 'rotate_180' in config:
                self.chk_rotate.setChecked(config['rotate_180'])
            return True
        except:
            return False

    def auto_load_last_config(self):
        if not self.load_input_config():
            return
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
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread = None
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.pause_playback()
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("选择输入源")
        msg_box.setText("请选择输入源类型：")
        btn_video = msg_box.addButton("视频文件", QMessageBox.ActionRole)
        btn_camera = msg_box.addButton("USB摄像头", QMessageBox.ActionRole)
        msg_box.addButton("取消", QMessageBox.RejectRole)
        msg_box.exec_()
        if msg_box.clickedButton() == btn_video:
            self.input_mode = 'video'
            self.select_files_for_video()
        elif msg_box.clickedButton() == btn_camera:
            self.input_mode = 'camera'
            self.select_files_for_camera()

    def select_files_for_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                     "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if not video_path:
            return
        self.video_path = video_path
        if not self.select_common_configs():
            return
        self.lbl_camera.setVisible(False)
        self.spin_camera_idx.setVisible(False)
        self.chk_rotate.setVisible(False)
        self.slider_progress.setEnabled(True)
        self.load_all_configs()
        self.save_input_config()

    def select_files_for_camera(self):
        if not self.select_common_configs():
            return
        self.lbl_camera.setVisible(True)
        self.spin_camera_idx.setVisible(True)
        self.chk_rotate.setVisible(True)
        self.slider_progress.setEnabled(False)
        self.load_configs_for_camera()
        self.save_input_config()

    def select_common_configs(self):
        calib_dir = QFileDialog.getExistingDirectory(self, "选择标定参数文件夹", "")
        if not calib_dir:
            return False
        self.calib_dir = calib_dir
        params_path, _ = QFileDialog.getOpenFileName(self, "选择参数文件", "", "JSON (*.json)")
        self.params_json_path = params_path if params_path else None
        data_dir = QFileDialog.getExistingDirectory(self, "选择data文件夹", "")
        if not data_dir:
            return False
        self.data_save_dir = data_dir
        return True

    def load_all_configs(self):
        if not self.load_calibration():
            return
        self.load_detector()
        if not self.load_video():
            return
        self.process_first_frame()

    def load_configs_for_camera(self):
        if not self.load_calibration():
            return
        self.load_detector()
        if not self.load_reference_data():
            return
        self.start_camera()

    def load_calibration(self):
        try:
            self.stereo_params = self.load_stereo_params(self.calib_dir)
            self.validate_params(self.stereo_params)
            self.calib_loaded = True
            self.status_bar.showMessage(f"标定参数已加载")
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

    def validate_params(self, params):
        assert params['K1'].shape == (3, 3)
        assert params['K2'].shape == (3, 3)
        assert params['D1'].shape == (1, 5)
        assert params['R'].shape == (3, 3)
        assert params['T'].size == 3

    def load_detector(self):
        config_path = self.params_json_path if self.params_json_path else "marker_params.json"
        self.detector = CircleDetector(config_path=config_path, verbose=True)

    def load_video(self):
        if not self.video_path:
            return False
        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开视频")
            return False
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 60
        self.slider_progress.setRange(0, self.total_frames - 1)
        self.spin_speed.setValue(int(self.fps))
        self.status_bar.showMessage(f"视频已加载: {self.total_frames}帧")
        return True

    def load_reference_data(self):
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
            QMessageBox.warning(self, "警告", f"未找到ROI文件")
            return False

        mirror_axis = 640
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
            except:
                pass

        result_dir = os.path.join(self.data_save_dir, "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")
        if not os.path.exists(first_frame_file):
            QMessageBox.warning(self, "警告", f"未找到首帧数据文件")
            return False

        try:
            data = np.genfromtxt(first_frame_file, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
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
        # 任务1: 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix
        return True

    def start_camera(self):
        self.camera_index = self.spin_camera_idx.value()
        self.rotate_180 = self.chk_rotate.isChecked()
        self.process_thread = FrameProcessThread()
        self.process_thread.set_processor(self.process_frame)
        self.process_thread.result_signal.connect(self.on_process_result)
        self.process_thread.start()
        self.camera_thread = CameraThread(self.camera_index, self.spin_speed.value(), self.rotate_180)
        self.camera_thread.frame_signal.connect(self.on_camera_frame)
        self.camera_thread.start()
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("暂停")
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.is_playing = True
        self.status_bar.showMessage(f"摄像头已启动")

    def on_camera_frame(self, frame, timestamp=0.0):
        if not self.FRAME_DATA['initialized']:
            return
        self.latest_camera_frame = frame
        self.current_frame_idx += 1
        h, w = frame.shape[:2]
        if self.FRAME_DATA['mirror_axis'] != w // 2:
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            self.FRAME_DATA['P1'] = P1
            self.FRAME_DATA['P2'] = P2
        if self.current_frame_idx % 10 == 0:
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx}")
        if self.is_playing and self.process_thread:
            self.process_thread.add_frame(frame.copy(), timestamp)
        else:
            self.display_frame(frame)

    def on_process_result(self, frame, points_3d, left_pts, right_pts, left_lost, right_lost, timestamp=0.0):
        if points_3d is not None:
            self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)
            if self.is_dragging or time.time() - self.drag_release_time < self.drag_cooldown:
                return
            current_time = time.time()
            if current_time - self.last_3d_update_time >= self.min_3d_update_interval:
                self.update_3d_view(points_3d)
                self.last_3d_update_time = current_time

    def process_first_frame(self):
        if not self.video_cap or not self.calib_loaded:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        mirror_axis = w // 2
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
            except:
                pass
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
            except:
                return
        result_dir = os.path.join(self.data_save_dir, "..", "result")
        first_frame_file = os.path.join(result_dir, "frame_000_points.txt")
        if not os.path.exists(first_frame_file):
            return
        try:
            data = np.genfromtxt(first_frame_file, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            points_3d = data[:, :3].astype(np.float64)
            pts1_R = data[:, 3:5].astype(np.float32)
            pts2_R = data[:, 5:7].astype(np.float32)
        except:
            return
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']
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
        # 任务1: 提取基准帧Delaunay拓扑边（平面图约束用）
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix
        self.current_frame_idx = 0
        self.display_frame(frame, pts1_R, pts2_R)
        self.update_3d_view(points_3d)
        self.btn_play_pause.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.status_bar.showMessage(f"首帧处理完成")

    def reset_first_frame(self):
        self.current_frame_idx = 0
        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
        if self.input_mode == 'video':
            self.slider_progress.setValue(0)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame, self.FRAME_DATA['left_points_0_R'], self.FRAME_DATA['right_points_0_R'])
                self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        else:
            self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        self.status_bar.showMessage("已重置参考点")
        self.consecutive_abnormal_frames = 0  # 重置后清零连续异常帧计数器

    def set_current_as_base(self):
        if not self.FRAME_DATA['initialized']:
            return
        left_pts = self.FRAME_DATA.get('left_points_0_pre')
        right_pts = self.FRAME_DATA.get('right_points_0_pre')
        if left_pts is None or right_pts is None:
            return
        self.FRAME_DATA['left_points_0_R'] = left_pts.copy()
        self.FRAME_DATA['right_points_0_R'] = right_pts.copy()
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R, T = self.stereo_params['R'], self.stereo_params['T']
        w, h = self.FRAME_DATA['mirror_axis'] * 2, 480
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
        l_pts_ud = cv2.undistortPoints(left_pts, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_pts, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)
        self.FRAME_DATA['base_3d_points'] = points_3d
        # 任务1: 更新基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(left_pts.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(right_pts.astype(np.float64))
        self.update_3d_view(points_3d)
        self.status_bar.showMessage(f"已设置为新基准帧")
        self.consecutive_abnormal_frames = 0  # 设为新基准帧后清零连续异常帧计数器

    def process_frame(self, frame):
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

        # ── 任务3: 平面图约束 Mesh Untangling ──────────────────────────────
        left_edges = self.FRAME_DATA.get('left_edges') or []
        right_edges = self.FRAME_DATA.get('right_edges') or []

        def find_crossing_pairs(pts, edges):
            """返回所有发生交叉的边对列表 [(e1_idx, e2_idx), ...]
            预计算所有边的 AABB，避免在内层循环重复提取坐标。
            """
            crossings = []
            n = len(edges)
            if n == 0:
                return crossings
            # 预计算每条边的 AABB: [min_x, max_x, min_y, max_y]
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
                    # 共享端点的边不检测
                    if a == c or a == d or b == c or b == d:
                        continue
                    # AABB 快速排斥（使用预计算结果）
                    if (bboxes[i, 1] < bboxes[j, 0] or bboxes[j, 1] < bboxes[i, 0] or
                            bboxes[i, 3] < bboxes[j, 2] or bboxes[j, 3] < bboxes[i, 2]):
                        continue
                    if self.segments_intersect(p1, p2, pts[c], pts[d]):
                        crossings.append((i, j))
            return crossings

        def untangle(pts_R, lost_mask, edges, matched_pairs_ref, side):
            """对单侧（左或右）执行解结，直接修改 pts_R。"""
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
                        # 如果任一端点是丢失点 → 强制拉回，禁止互换
                        p_lost = lost_mask[p]
                        q_lost = lost_mask[q]
                        if p_lost or q_lost:
                            # ═══ 弹性拉回丢失点：使用安全邻居位移中位数预测 ═══
                            for lost_idx in ([p] if p_lost else []) + ([q] if q_lost else []):
                                # 找该点在拓扑边中的邻居
                                neighbors = []
                                for ea, eb in edges:
                                    if ea == lost_idx and not lost_mask[eb]:
                                        neighbors.append(eb)
                                    elif eb == lost_idx and not lost_mask[ea]:
                                        neighbors.append(ea)
                                # 防爆保护：无邻居则跳过，避免后续计算出错
                                if not neighbors:
                                    continue
                                # 检查邻居是否参与交叉（只用未交叉邻居）
                                crossing_nodes = set()
                                for ci, cj in crossings:
                                    for nd in edges[ci] + edges[cj]:
                                        crossing_nodes.add(nd)
                                safe_neighbors = [n for n in neighbors if n not in crossing_nodes]
                                # 防爆保护：无安全邻居则跳过，禁止对空数组调用 median
                                if not safe_neighbors:
                                    continue
                                # 使用安全邻居位移中位数预测丢失点位置
                                base_pts = self.FRAME_DATA['left_points_0_R'] if side == 'left' \
                                    else self.FRAME_DATA['right_points_0_R']
                                moves = [pts_R[n] - base_pts[n] for n in safe_neighbors]
                                median_move = np.median(moves, axis=0)
                                # 弹性外推：50% 保留原外推惯性 + 50% 拉向内侧安全位置
                                safe_pos = base_pts[lost_idx] + median_move
                                pts_R[lost_idx] = 0.5 * pts_R[lost_idx] + 0.5 * safe_pos
                            resolved_any = True
                            break

                        # 尝试互换 p 和 q
                        pts_R_trial = pts_R.copy()
                        pts_R_trial[p], pts_R_trial[q] = pts_R[q].copy(), pts_R[p].copy()
                        new_crossings = find_crossing_pairs(pts_R_trial, edges)
                        new_cross_set = set(map(tuple, new_crossings))
                        old_cross_set = set(map(tuple, crossings))
                        # 互换有效：消除了当前交叉且没有引入新交叉
                        if (ei, ej) not in new_cross_set and len(new_cross_set) < len(old_cross_set):
                            pts_R[p], pts_R[q] = pts_R_trial[p].copy(), pts_R_trial[q].copy()
                            resolved_any = True
                            break
                    if resolved_any:
                        break
                if not resolved_any:
                    break

        # ── 灾难性熔断检测 ────────────────────────────────────────────────
        # 在调用 untangle 之前，先评估当前帧的混乱程度
        # 若丢失点比例过高或交叉边数量爆炸，直接判定为灾难性异常，跳过 untangle 以防卡死
        num_total = len(pre_l)
        num_lost = int(np.sum(left_lost_mask) + np.sum(right_lost_mask))
        lost_ratio = num_lost / (num_total * 2) if num_total > 0 else 1.0
        left_initial_crossings = find_crossing_pairs(left_points_R, left_edges) if left_edges else []
        right_initial_crossings = find_crossing_pairs(right_points_R, right_edges) if right_edges else []
        is_catastrophic = (lost_ratio > 0.5 or
                           len(left_initial_crossings) > 20 or
                           len(right_initial_crossings) > 20)

        if is_catastrophic:
            # 灾难性异常：直接跳过 untangle 和 3D 计算，进入异常拦截
            print(f"[灾难性熔断 #{self.current_frame_idx}] "
                  f"lost_ratio={lost_ratio:.2f}, "
                  f"left_crossings={len(left_initial_crossings)}, "
                  f"right_crossings={len(right_initial_crossings)}")
            self.FRAME_DATA['current_left_crossings'] = left_initial_crossings
            self.FRAME_DATA['current_right_crossings'] = right_initial_crossings
        else:
            # 正常路径：执行 untangle 解结并计算 3D 坐标
            if left_edges:
                untangle(left_points_R, left_lost_mask, left_edges, matched_pairs, 'left')
            if right_edges:
                untangle(right_points_R, right_lost_mask, right_edges, matched_pairs, 'right')
            # 保存最终交叉结果供 display_frame 直接读取，避免 UI 线程重复计算
            self.FRAME_DATA['current_left_crossings'] = (
                find_crossing_pairs(left_points_R, left_edges) if left_edges else [])
            self.FRAME_DATA['current_right_crossings'] = (
                find_crossing_pairs(right_points_R, right_edges) if right_edges else [])
        # ── Mesh Untangling 结束 ────────────────────────────────────────────

        # 灾难性熔断时跳过 3D 计算，直接进入异常处理
        if not is_catastrophic:
            K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
            K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
            R, T = self.stereo_params['R'], self.stereo_params['T']
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
            l_pts_ud = cv2.undistortPoints(left_points_R, K1, D1, R=R1, P=P1).squeeze()
            r_pts_ud = cv2.undistortPoints(right_points_R, K2, D2, R=R2, P=P2).squeeze()
            points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

            # 常规异常检测（Z轴突变 / 点间距异常）
            base_3d = self.FRAME_DATA['base_3d_points']
            is_abnormal = False
            abnormal_reason = []
            if base_3d is not None and len(points_3d) == len(base_3d):
                z_diff = points_3d[:, 2] - base_3d[:, 2]
                if np.any(z_diff > 1.0):
                    is_abnormal = True
                    max_z_diff = np.max(np.abs(z_diff))
                    abnormal_reason.append(f"Z轴深度异常: 最大偏移 {max_z_diff:.3f}mm (阈值: 1.0mm)")
                if not is_abnormal and len(points_3d) >= 2:
                    from scipy.spatial.distance import pdist
                    distances = pdist(points_3d)
                    if np.any(distances < 0.3):
                        is_abnormal = True
                        min_dist = np.min(distances)
                        abnormal_reason.append(f"点间距异常: 最小距离 {min_dist:.3f}mm (阈值: 0.3mm)")
        else:
            is_abnormal = True
            abnormal_reason = ["灾难性熔断"]
            base_3d = self.FRAME_DATA['base_3d_points']

        # ── 连续异常帧计数与分级处理 ──────────────────────────────────────
        if not is_abnormal:
            # 正常帧：清零计数器，更新状态，返回结果
            self.consecutive_abnormal_frames = 0
            self.FRAME_DATA['left_points_0_pre'] = left_points_R
            self.FRAME_DATA['right_points_0_pre'] = right_points_R
            return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask

        # 异常帧处理
        self.consecutive_abnormal_frames += 1
        print(f"[异常帧 #{self.current_frame_idx}] {' | '.join(abnormal_reason)} "
              f"（连续第 {self.consecutive_abnormal_frames} 帧）")

        if self.consecutive_abnormal_frames >= 5:
            # 连续 5 帧异常 → 强制重置：将 _pre 覆写为首帧基准点 _R，重启追踪
            print(f"[强制重置] 连续异常已达 {self.consecutive_abnormal_frames} 帧，强制重置至首帧基准")
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
            # 时光停滞：保持 _pre 不变，返回上一帧良好数据
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
        if mask is None:
            return points
        valid_points = []
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        height, width = mask.shape[:2]
        for (x, y, r) in points:
            x_int, y_int = int(round(x)), int(round(y))
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
            A = np.array([x1 * P1[2, :] - P1[0, :], y1 * P1[2, :] - P1[1, :],
                          x2 * P2[2, :] - P2[0, :], y2 * P2[2, :] - P2[1, :]])
            _, _, V = np.linalg.svd(A)
            X_homo = V[-1, :]
            X = X_homo[:3] / X_homo[3]
            points_3d[i] = X
        return points_3d

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
            n_ref, n_det = len(ref_pts), len(det_pts)
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

    @staticmethod
    def extract_edges(points_2d):
        """从2D点集提取Delaunay三角剖分的去重边列表，并过滤幽灵长边。
        过滤规则：只保留长度 <= 中位数边长 * 1.3 的边（收紧阈值以更严格剔除凹陷边缘的错误连线）。
        返回: [(id_a, id_b), ...] 其中 id_a < id_b。
        """
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
        # 中位数长度过滤：剔除跨越空白区域的幽灵长边（阈值从1.5收紧至1.3）
        edge_list = list(edges)
        lengths = np.array([np.linalg.norm(pts[a] - pts[b]) for a, b in edge_list])
        threshold = np.median(lengths) * 1.3
        return [e for e, l in zip(edge_list, lengths) if l <= threshold]

    @staticmethod
    def segments_intersect(p1, p2, p3, p4):
        """判断线段 p1p2 与线段 p3p4 是否真正相交（不含共线/端点接触）。
        先用 AABB 快速排斥，再用带容差的向量叉积 CCW 跨立实验。
        eps=3.0 增加对亚像素抖动的包容度，减少边缘透视畸变引起的假交叉。
        """
        # AABB 快速排斥
        if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or
                max(p3[0], p4[0]) < min(p1[0], p2[0]) or
                max(p1[1], p2[1]) < min(p3[1], p4[1]) or
                max(p3[1], p4[1]) < min(p1[1], p2[1])):
            return False
        # 带容差的 CCW 叉积跨立实验（容差从2.0放大至3.0）
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

    def display_frame(self, frame, left_pts=None, right_pts=None, left_lost=None, right_lost=None):
        display_img = frame.copy()

        # 绘制Delaunay拓扑网格连线（交叉结果直接读取缓存，不重复计算）
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
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        label_size = self.lbl_video.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)
        if self.input_mode == 'video':
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx} / {self.total_frames - 1}")

    def update_3d_view(self, points_3d):
        if self.is_dragging or time.time() - self.drag_release_time < self.drag_cooldown:
            return
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')
        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)
            points[:, 1] = -points[:, 1]
            base_local = self.transform_to_local_coordinates(self.FRAME_DATA['base_3d_points'])
            base_local[:, 1] = -base_local[:, 1]
            displacement_vectors = points - base_local
            deformation = np.linalg.norm(displacement_vectors, axis=1)
            self.avg_displacement_vector = np.mean(displacement_vectors, axis=0)
            self.avg_displacement_magnitude = np.mean(deformation)
            self.max_displacement = np.max(deformation)
            self.min_displacement = np.min(deformation)
            self.current_points_3d = points
            sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=deformation, cmap='jet', s=50, vmin=0, vmax=1.5)
            cbar = self.fig_3d.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label('Deformation (mm)', rotation=270, labelpad=15)
            if self.avg_displacement_magnitude > 0.01 and self.chk_show_avg_disp.isChecked():
                center = np.mean(points, axis=0)
                arrow_scale = 5.0
                ax.quiver(center[0], center[1], center[2],
                         self.avg_displacement_vector[0] * arrow_scale,
                         self.avg_displacement_vector[1] * arrow_scale,
                         self.avg_displacement_vector[2] * arrow_scale,
                         color='red', arrow_length_ratio=0.3, linewidth=2.5,
                         label=f'Avg: {self.avg_displacement_magnitude:.3f}mm')
                ax.legend(loc='upper left', fontsize=8)
        else:
            points = points_3d.copy()
            points[:, 1] = -points[:, 1]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)
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
        self.ax_3d = ax
        self.canvas_3d.draw()

    def update_max_dist(self, value):
        self.max_match_dist = value

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        self.is_playing = True
        self.btn_play_pause.setText("暂停")
        if self.input_mode == 'video':
            interval = int(1000 / self.spin_speed.value())
            self.play_timer.start(interval)
        self.status_bar.showMessage("播放中...")

    def pause_playback(self):
        self.is_playing = False
        self.btn_play_pause.setText("播放")
        if self.input_mode == 'video':
            self.play_timer.stop()
        self.status_bar.showMessage("已暂停")

    def update_playback_speed(self, fps):
        if self.input_mode == 'video' and self.is_playing:
            interval = int(1000 / fps)
            self.play_timer.setInterval(interval)
        elif self.input_mode == 'camera' and self.camera_thread:
            self.camera_thread.target_fps = fps

    def on_timer_tick(self):
        if self.input_mode != 'video':
            return
        if self.current_frame_idx >= self.total_frames - 1:
            self.current_frame_idx = 0
            self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
            self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.video_cap.read()
        if not ret:
            self.pause_playback()
            return
        if self.current_frame_idx == 0:
            self.display_frame(frame, self.FRAME_DATA['left_points_0_R'], self.FRAME_DATA['right_points_0_R'])
            self.update_3d_view(self.FRAME_DATA['base_3d_points'])
        else:
            points_3d, left_pts, right_pts, left_lost, right_lost = self.process_frame(frame)
            if points_3d is not None:
                self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)
                self.update_3d_view(points_3d)
        self.slider_progress.blockSignals(True)
        self.slider_progress.setValue(self.current_frame_idx)
        self.slider_progress.blockSignals(False)
        self.current_frame_idx += 1

    def on_slider_pressed(self):
        if self.input_mode != 'video':
            return
        self.slider_dragging = True
        if self.is_playing:
            self.play_timer.stop()

    def on_slider_released(self):
        if self.input_mode != 'video':
            return
        self.slider_dragging = False
        self.seek_to_frame(self.slider_progress.value())
        if self.is_playing:
            interval = int(1000 / self.spin_speed.value())
            self.play_timer.start(interval)

    def on_slider_changed(self, value):
        if self.input_mode != 'video':
            return
        if self.slider_dragging:
            self.lbl_frame_info.setText(f"帧: {value} / {self.total_frames - 1}")

    def seek_to_frame(self, frame_idx):
        if self.input_mode != 'video':
            return
        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
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

    def on_3d_mouse_press(self, event):
        self.is_dragging = True
        self._was_playing_before_drag = self.is_playing
        if self.is_playing:
            if self.input_mode == 'video':
                self.play_timer.stop()
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
                if self.input_mode == 'video':
                    interval = int(1000 / self.spin_speed.value())
                    self.play_timer.start(interval)
                self._was_playing_before_drag = False

    def closeEvent(self, event):
        self.play_timer.stop()
        if self.process_thread:
            self.process_thread.running = False
            self.process_thread.has_new_frame.set()
        if self.camera_thread:
            self.camera_thread.running = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if self.process_thread:
            self.process_thread.stop()
            self.process_thread = None
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)
    window = VideoPointCloudPlayer()
    window.show()
    sys.exit(app.exec_())
