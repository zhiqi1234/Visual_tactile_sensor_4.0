# -*- coding: utf-8 -*-
'''
视频/摄像头标记点跟踪与三维重建播放器
功能说明：
1. 支持读取视频文件或USB摄像头实时采集
2. 逐帧进行标记点检测和三维重建
3. 启动前弹出文件选择窗口：标定文件夹、参数JSON文件、首帧ROI和匹配数据保存文件夹
4. GUI中同时显示视频帧和三维点云
5. 支持暂停/播放，支持手动调整三维点云观察角度
6. 支持"设置为首帧"动态更新基准帧
7. 六维力传感器实时订阅与显示
8. HDF5数据采集：同步记录视觉3D坐标(xyz/dxyz)和六维力传感器数据(fx,fy,fz,mx,my,mz)
9. 压电传感器采集：串口读取全部8通道原始数据，实时波形显示，与视觉同步保存到HDF5
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
from collections import deque
from datetime import datetime
import numpy as np
import h5py
import cv2
import serial
import serial.tools.list_ports
import topic # type: ignore
import message # type: ignore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QGroupBox, QStatusBar, QSlider, QSpinBox, QSplitter,
                             QCheckBox, QComboBox)
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


class PiezoSerialThread(QThread):
    """压电传感器串口采集线程
    - 采集线程内部维护缓冲区，不通过高频 signal 打断主线程
    - 按ADC组(0-4)分别存储，供主线程按组选择显示/保存
    - 每积累 plot_interval 帧才发一次低频 plot_ready 信号刷新波形
    """
    plot_ready = pyqtSignal()  # 低频刷新信号，约5-10Hz

    # 帧格式常量
    _FRAME_LEN = 29
    _REF_V = 3.3
    _MAX_VAL = 2 ** 23
    _N_GROUPS = 5

    def __init__(self, port='COM1', baudrate=921600, plot_interval=50):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self._serial = None
        self._rx_buf = b''
        self._last_clear_time = 0

        # 按ADC组分组的共享缓冲区：5组，每行 (timestamp, ch0..ch7)
        # 预览窗口只保留最近5000帧用于波形显示
        self._group_bufs = [deque(maxlen=5000) for _ in range(self._N_GROUPS)]
        self._buf_lock = threading.Lock()

        # 当前选中的ADC组和预览通道（主线程可随时改，int赋值原子安全）
        self._selected_adc_group = 0
        self._preview_ch = 0

        # 节流计数
        self._plot_counter = 0
        self._plot_interval = plot_interval

        # 流式录制：采集期间直接追加写入，避免长时间采集内存溢出
        self._recording = False             # 是否正在录制
        self._record_bufs = [[] for _ in range(self._N_GROUPS)]  # 每组的待刷新缓冲
        self._record_lock = threading.Lock()
        self._record_filepath = None        # 当前录制文件路径（由主线程设置）
        self._record_hdf_handle = None      # 打开的h5py文件句柄
        self._record_ds_ts = [None] * self._N_GROUPS   # 各组 timestamp dataset
        self._record_ds_val = [None] * self._N_GROUPS  # 各组 values dataset
        self._flush_interval = 200          # 每积累多少帧就刷新一次到HDF5
        self._flush_counter = 0

    # ── 主线程调用的接口 ──────────────────────────────────

    def set_adc_group(self, group):
        """切换ADC组（0-4），无需加锁"""
        self._selected_adc_group = max(0, min(self._N_GROUPS - 1, group))

    def set_preview_channel(self, ch):
        """切换波形预览通道（0-7），无需加锁"""
        self._preview_ch = max(0, min(7, ch))

    def get_plot_data(self):
        """获取当前选中ADC组、预览通道的波形数据（用于 PyQtGraph 刷新）
        返回 np.ndarray 电压序列，最多5000点
        """
        with self._buf_lock:
            buf = list(self._group_bufs[self._selected_adc_group])
        if not buf:
            return np.array([], dtype=np.float32)
        ch = self._preview_ch + 1  # buf 每行: (ts, ch0..ch7)，索引+1
        return np.array([row[ch] for row in buf[-5000:]], dtype=np.float32)

    def get_recording_snapshot(self):
        """获取当前选中ADC组的完整缓冲区快照用于保存
        timestamps: (N,) float64
        values:     (N, 8) float32
        """
        with self._buf_lock:
            buf = list(self._group_bufs[self._selected_adc_group])
        if not buf:
            return np.array([], dtype=np.float64), np.zeros((0, 8), dtype=np.float32)
        timestamps = np.array([row[0] for row in buf], dtype=np.float64)
        values = np.array([row[1:] for row in buf], dtype=np.float32)  # (N, 8)
        return timestamps, values

    def clear_buffer(self):
        """采集开始时清空所有ADC组缓冲区"""
        with self._buf_lock:
            for buf in self._group_bufs:
                buf.clear()

    # ── 线程主循环 ────────────────────────────────────────

    def run(self):
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self._serial.reset_input_buffer()
            self.running = True
            self._last_clear_time = time.time()

            while self.running:
                if self._serial.in_waiting > 0:
                    self._rx_buf += self._serial.read(self._serial.in_waiting)

                    # 防积压：缓冲区过大时丢弃旧数据
                    now = time.time()
                    if len(self._rx_buf) > 5000 or (
                            now - self._last_clear_time > 2.0 and len(self._rx_buf) > 1000):
                        self._rx_buf = self._rx_buf[-1000:]
                        self._last_clear_time = now

                    self._process_frames()
                else:
                    time.sleep(0.0005)  # 无数据时短暂休眠，避免空转
        except serial.SerialException as e:
            print(f"[PiezoSerialThread] 串口错误: {e}")

    def stop(self):
        self.running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.quit()
        self.wait()

    def _process_frames(self):
        """从接收缓冲区解析完整帧，写入共享缓冲区"""
        while True:
            idx = self._rx_buf.find(b'\xaa\xaa')
            if idx == -1 or len(self._rx_buf) < idx + self._FRAME_LEN:
                break

            tail = self._rx_buf[idx + self._FRAME_LEN - 2: idx + self._FRAME_LEN]
            if tail != b'\xff\xff':
                self._rx_buf = self._rx_buf[idx + 2:]
                continue

            payload = self._rx_buf[idx + 2: idx + self._FRAME_LEN - 2]
            self._rx_buf = self._rx_buf[idx + self._FRAME_LEN:]

            # payload[0] = ADC组标识（0-4），payload[1:] = 8通道×3字节
            adc_group = payload[0]
            if adc_group >= self._N_GROUPS:
                continue
            data_bytes = payload[1:]
            if len(data_bytes) < 24:
                continue

            ts = time.time()
            voltages = []
            for ch in range(8):
                raw = data_bytes[ch * 3: ch * 3 + 3]
                val = self._bytes_to_decimal(raw)
                v = (val / self._MAX_VAL) * self._REF_V
                if ch == 0 or ch == 1:
                    v = -v
                voltages.append(v)

            with self._buf_lock:
                self._group_bufs[adc_group].append((ts, *voltages))  # (ts, v0..v7)

            # 流式录制：采集期间将数据追加到待刷新缓冲
            if self._recording:
                with self._record_lock:
                    self._record_bufs[adc_group].append((ts, *voltages))
                self._flush_counter += 1
                if self._flush_counter >= self._flush_interval:
                    self._flush_counter = 0
                    self._flush_to_hdf5()

            # 节流：每 _plot_interval 帧发一次刷新信号
            self._plot_counter += 1
            if self._plot_counter >= self._plot_interval:
                self._plot_counter = 0
                self.plot_ready.emit()

    def start_streaming_record(self, filepath):
        """开始流式录制，打开HDF5文件句柄（由主线程在采集开始时调用）"""
        try:
            f = h5py.File(filepath, 'a')  # 追加模式，文件可能已有其他组
            self._record_hdf_handle = f
            self._record_ds_ts = [None] * self._N_GROUPS
            self._record_ds_val = [None] * self._N_GROUPS
            # 预先在文件中创建可扩展的压电数据集
            if 'piezo_stream' not in f:
                pg = f.create_group('piezo_stream')
                pg.attrs['n_channels'] = 8
                pg.attrs['channel_names'] = 'CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8'
                pg.attrs['unit'] = 'V'
            pg = f['piezo_stream']
            for g in range(self._N_GROUPS):
                grp_name = f'adc{g+1}'
                if grp_name not in pg:
                    sub = pg.create_group(grp_name)
                    sub.create_dataset('timestamp', shape=(0,), maxshape=(None,),
                                       dtype=np.float64, chunks=(1000,))
                    sub.create_dataset('values', shape=(0, 8), maxshape=(None, 8),
                                       dtype=np.float32, chunks=(1000, 8))
                self._record_ds_ts[g] = pg[grp_name]['timestamp']
                self._record_ds_val[g] = pg[grp_name]['values']
            # 清空待刷新缓冲
            with self._record_lock:
                self._record_bufs = [[] for _ in range(self._N_GROUPS)]
            self._flush_counter = 0
            self._record_filepath = filepath
            self._recording = True
            print(f"[PiezoSerialThread] 流式录制已开始: {filepath}")
        except Exception as e:
            print(f"[PiezoSerialThread] 流式录制启动失败: {e}")
            self._recording = False

    def stop_streaming_record(self):
        """停止流式录制，刷新剩余数据并关闭文件（由主线程在采集停止时调用）"""
        self._recording = False
        # 最终刷新剩余数据
        self._flush_to_hdf5(final=True)
        if self._record_hdf_handle is not None:
            try:
                self._record_hdf_handle.close()
            except Exception:
                pass
            self._record_hdf_handle = None
        print("[PiezoSerialThread] 流式录制已停止")

    def _flush_to_hdf5(self, final=False):
        """将各组待刷新缓冲的数据追加写入HDF5（在采集线程内调用）"""
        if self._record_hdf_handle is None:
            return
        for g in range(self._N_GROUPS):
            with self._record_lock:
                rows = self._record_bufs[g]
                if not rows:
                    continue
                self._record_bufs[g] = []
            # 转换
            ts_arr = np.array([r[0] for r in rows], dtype=np.float64)
            val_arr = np.array([r[1:] for r in rows], dtype=np.float32)
            # 扩展 dataset
            ds_ts = self._record_ds_ts[g]
            ds_val = self._record_ds_val[g]
            if ds_ts is None or ds_val is None:
                continue
            try:
                n_old = ds_ts.shape[0]
                n_new = len(ts_arr)
                ds_ts.resize(n_old + n_new, axis=0)
                ds_ts[n_old:] = ts_arr
                ds_val.resize(n_old + n_new, axis=0)
                ds_val[n_old:] = val_arr
                if final:
                    self._record_hdf_handle.flush()
            except Exception as e:
                print(f"[PiezoSerialThread] HDF5写入失败 ADC{g+1}: {e}")

    @staticmethod
    def _bytes_to_decimal(data):
        b1, b2, b3 = data
        v = (b1 << 16) | (b2 << 8) | b3
        if v & 0x800000:
            v -= 0x1000000
        return v


class CameraThread(QThread):
    """摄像头采集线程"""
    frame_signal = pyqtSignal(np.ndarray, float)
    fps_signal = pyqtSignal(float)  # 实时采集帧率

    def __init__(self, camera_index=0, fps=60, rotate_180=True):
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
        print(f"[Camera] 分辨率={actual_w}x{actual_h}, FOURCC={actual_fourcc:08x}, "
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
        # 给线程一个超时时间，避免 cv2.read() 阻塞导致无限等待
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


class FrameProcessThread(QThread):
    """帧处理线程 - 将耗时的检测和重建放到后台"""
    result_signal = pyqtSignal(object, object, object, object, object, object, object, object)  # frame, points_3d, left_pts, right_pts, left_lost, right_lost, timestamp, is_abnormal

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
            # 只保留最新帧，丢弃旧帧
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
                    self.result_signal.emit(frame, points_3d, left_pts, right_pts, left_lost, right_lost, timestamp, is_abnormal)
                except Exception as e:
                    pass

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


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


class VideoPointCloudPlayer(QMainWindow):
    """视频/摄像头点云播放器主窗口"""

    # 配置文件路径（保存上次选择的输入源配置）
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_input_config.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉系统 - 标记点跟踪与三维重建")
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

        # PCA点云是否需要绕Z轴旋转180°（自动检测）
        self.pca_flip_xy = False

        # FPS 显示
        self._fps_label = QLabel("FPS: --")

        # 检测器
        self.detector = None

        # 匹配参数
        self.max_match_dist = 50
        self.consecutive_abnormal_frames = 0

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

        # 位移统计数据
        self.avg_displacement_vector = np.array([0.0, 0.0, 0.0])  # 平均位移向量
        self.avg_displacement_magnitude = 0.0  # 平均位移大小
        self.max_displacement = 0.0  # 最大位移
        self.min_displacement = 0.0  # 最小位移
        self.current_points_3d = None  # 当前帧3D点

        # 图像缓冲（预分配，避免每帧 np.full 分配）
        self._left_img_buf = None
        self._right_img_buf = None
        # KDTree 缓存（pre 数组不变时复用）
        self._pre_l_tree = None
        self._pre_l_tree_id = None
        self._pre_r_tree = None
        self._pre_r_tree_id = None

        # HDF5 数据采集
        self.is_recording = False
        self.recording_buffer = {
            'timestamps': [],
            'xyz': [],
            'abnormal': [],  # 异常帧标记: True=异常帧(点追踪故障重置)
        }

        # 六维力传感器相关
        self.ft_node = None
        self.ft_subscription = None
        self.ft_lock = threading.Lock()
        self.latest_ft_data = None  # 最新的力传感器数据 (list of FtvalueInfo)
        self.latest_ft_timestamp = 0.0  # 最新力传感器数据的时间戳
        self.ft_recording_buffer = {
            'timestamps': [],
            'ft_values': [],  # 每个元素为 [fx, fy, fz, mx, my, mz]
        }

        # 压电传感器相关
        self.piezo_thread = None          # PiezoSerialThread 实例
        self.piezo_preview_ch = 0         # 当前波形预览通道（0-7）
        self.piezo_adc_group = 0          # 当前选中ADC组（0-4）
        # 采集时记录的起始时间戳，用于截取本次采集段
        self._piezo_record_start_ts = 0.0

        # 初始化界面
        self.init_ui()

        # 启动六维力传感器订阅
        self.start_ft_subscription()

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

        self.btn_record = QPushButton("开始采集")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)

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
        self.chk_rotate.setChecked(False)

        control_layout.addWidget(self.btn_select_input)
        control_layout.addWidget(self.btn_play_pause)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.btn_set_as_base)
        control_layout.addWidget(self.btn_record)
        control_layout.addWidget(self.lbl_frame_info)
        control_layout.addWidget(self.spin_speed)
        control_layout.addWidget(self.spin_max_dist)
        control_layout.addWidget(self.lbl_camera)
        control_layout.addWidget(self.spin_camera_idx)
        control_layout.addWidget(self.chk_rotate)

        # ── 压电传感器控件 ──
        control_layout.addWidget(QLabel("  |  压电:"))

        self.cbb_piezo_port = QComboBox()
        self.cbb_piezo_port.setMinimumWidth(80)
        self.cbb_piezo_port.addItems(self._get_serial_ports())
        control_layout.addWidget(self.cbb_piezo_port)

        self.btn_piezo_refresh = QPushButton("刷新")
        self.btn_piezo_refresh.setMaximumWidth(40)
        self.btn_piezo_refresh.clicked.connect(self._refresh_piezo_ports)
        control_layout.addWidget(self.btn_piezo_refresh)

        self.cbb_piezo_channel = QComboBox()
        self.cbb_piezo_channel.addItems([f"CH{i+1}" for i in range(8)])
        self.cbb_piezo_channel.setCurrentIndex(0)
        self.cbb_piezo_channel.currentIndexChanged.connect(self._on_piezo_channel_changed)
        control_layout.addWidget(self.cbb_piezo_channel)

        self.cbb_piezo_adc_group = QComboBox()
        self.cbb_piezo_adc_group.addItems([f"ADC{i+1}" for i in range(5)])
        self.cbb_piezo_adc_group.setCurrentIndex(0)
        self.cbb_piezo_adc_group.currentIndexChanged.connect(self._on_piezo_adc_group_changed)
        control_layout.addWidget(self.cbb_piezo_adc_group)

        self.btn_piezo_connect = QPushButton("连接压电")
        self.btn_piezo_connect.clicked.connect(self._toggle_piezo_connection)
        control_layout.addWidget(self.btn_piezo_connect)

        control_layout.addStretch()

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

        # 下部：六维力传感器实时显示区域
        ft_group = QGroupBox("六维力传感器实时数据")
        ft_layout = QHBoxLayout()
        self.lbl_ft_fx = QLabel("Fx: --")
        self.lbl_ft_fy = QLabel("Fy: --")
        self.lbl_ft_fz = QLabel("Fz: --")
        self.lbl_ft_mx = QLabel("Mx: --")
        self.lbl_ft_my = QLabel("My: --")
        self.lbl_ft_mz = QLabel("Mz: --")
        for lbl in [self.lbl_ft_fx, self.lbl_ft_fy, self.lbl_ft_fz,
                     self.lbl_ft_mx, self.lbl_ft_my, self.lbl_ft_mz]:
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
            ft_layout.addWidget(lbl)
        ft_layout.addStretch()
        ft_group.setLayout(ft_layout)
        main_splitter.addWidget(ft_group)

        # 下部：压电信号实时波形（PyQtGraph）
        piezo_group = QGroupBox("压电信号实时波形")
        piezo_layout = QVBoxLayout(piezo_group)
        piezo_layout.setContentsMargins(4, 16, 4, 4)

        self.piezo_plot_widget = pg.GraphicsLayoutWidget()
        self.piezo_plot_widget.setBackground('w')
        self.piezo_plot_item = self.piezo_plot_widget.addPlot()
        self.piezo_plot_item.setLabel('left', '电压 (V)')
        self.piezo_plot_item.setLabel('bottom', '采样点')
        self.piezo_plot_item.setYRange(-3.3, 3.3)
        self.piezo_plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.piezo_plot_item.addLegend(offset=(10, 10))
        # 当前预览通道曲线（蓝色，粗）
        self.piezo_curve_preview = self.piezo_plot_item.plot(
            pen=pg.mkPen('#2196F3', width=2), name='预览通道')
        # 状态标签（右上角显示通道名和采样率）
        self.piezo_status_label = pg.LabelItem(
            text='未连接', color='#666666', size='10pt')
        self.piezo_plot_widget.addItem(self.piezo_status_label, row=0, col=1)

        piezo_layout.addWidget(self.piezo_plot_widget)
        main_splitter.addWidget(piezo_group)

        # 分割比例：视觉区 5 : 力显示 1 : 压电波形 2
        main_splitter.setSizes([500, 60, 160])
        main_layout.addWidget(main_splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self._fps_label)
        self.status_bar.showMessage("就绪 - 请选择视频和配置文件")

        # 3D视图鼠标事件
        self.canvas_3d.mpl_connect('button_press_event', self.on_3d_mouse_press)
        self.canvas_3d.mpl_connect('button_release_event', self.on_3d_mouse_release)
        self.canvas_3d.mpl_connect('motion_notify_event', self.on_3d_mouse_motion)

        # 力传感器数据刷新定时器
        self.ft_display_timer = QTimer(self)
        self.ft_display_timer.timeout.connect(self.update_ft_display)
        self.ft_display_timer.start(100)  # 每100ms刷新一次

    # ---- 六维力传感器 ----
    def start_ft_subscription(self):
        """启动六维力传感器数据订阅"""
        try:
            ft_options = topic.NodeOptions()
            ft_options.node_name = 'v5_ft_subscriber'
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
        """力传感器数据回调（在订阅线程中执行）"""
        parm = message.SystemStateData()
        message.display_rt(tt, parm)

        timestamp = time.time()

        with self.ft_lock:
            self.latest_ft_data = parm.controller.ftvalues
            self.latest_ft_timestamp = timestamp

            # 如果正在采集，将数据写入缓冲区
            if self.is_recording and parm.controller.ftvalues:
                for ftv in parm.controller.ftvalues:
                    self.ft_recording_buffer['timestamps'].append(timestamp)
                    self.ft_recording_buffer['ft_values'].append(
                        [ftv.fx, ftv.fy, ftv.fz, ftv.mx, ftv.my, ftv.mz])

    def update_ft_display(self):
        """定时刷新力传感器数据显示（在GUI线程中执行）"""
        with self.ft_lock:
            ft_data = self.latest_ft_data

        if ft_data and len(ft_data) > 0:
            ftv = ft_data[0]  # 显示第一个传感器
            self.lbl_ft_fx.setText(f"Fx: {ftv.fx:+.3f}")
            self.lbl_ft_fy.setText(f"Fy: {ftv.fy:+.3f}")
            self.lbl_ft_fz.setText(f"Fz: {ftv.fz:+.3f}")
            self.lbl_ft_mx.setText(f"Mx: {ftv.mx:+.3f}")
            self.lbl_ft_my.setText(f"My: {ftv.my:+.3f}")
            self.lbl_ft_mz.setText(f"Mz: {ftv.mz:+.3f}")

    # ---- 压电传感器 ----

    @staticmethod
    def _get_serial_ports():
        """获取当前可用串口列表"""
        try:
            return [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            return []

    def _refresh_piezo_ports(self):
        """刷新串口下拉列表"""
        current = self.cbb_piezo_port.currentText()
        self.cbb_piezo_port.clear()
        ports = self._get_serial_ports()
        self.cbb_piezo_port.addItems(ports)
        # 尽量保持原来选中的串口
        if current in ports:
            self.cbb_piezo_port.setCurrentText(current)

    def _on_piezo_channel_changed(self, index):
        """切换预览通道，通知采集线程（如果已连接）"""
        self.piezo_preview_ch = index
        if self.piezo_thread is not None:
            self.piezo_thread.set_preview_channel(index)
        ch_name = f"CH{index + 1}"
        self.piezo_plot_item.setTitle(f"压电信号 — ADC{self.piezo_adc_group + 1} {ch_name}")

    def _on_piezo_adc_group_changed(self, index):
        """切换ADC组，通知采集线程（如果已连接）"""
        self.piezo_adc_group = index
        if self.piezo_thread is not None:
            self.piezo_thread.set_adc_group(index)
        ch_name = f"CH{self.piezo_preview_ch + 1}"
        self.piezo_plot_item.setTitle(f"压电信号 — ADC{index + 1} {ch_name}")

    def _toggle_piezo_connection(self):
        """连接/断开压电传感器"""
        if self.piezo_thread is None or not self.piezo_thread.isRunning():
            port = self.cbb_piezo_port.currentText()
            if not port:
                QMessageBox.warning(self, "警告", "请先选择压电串口")
                return

            self.piezo_thread = PiezoSerialThread(port, baudrate=921600, plot_interval=50)
            self.piezo_thread.set_preview_channel(self.piezo_preview_ch)
            self.piezo_thread.set_adc_group(self.piezo_adc_group)
            # 只连接低频刷新信号，不走高频数据回调
            self.piezo_thread.plot_ready.connect(self._refresh_piezo_plot)
            self.piezo_thread.start()

            self.btn_piezo_connect.setText("断开压电")
            self.btn_piezo_connect.setStyleSheet("background-color: #ff7043; color: white;")
            ch_name = f"CH{self.piezo_preview_ch + 1}"
            self.piezo_plot_item.setTitle(f"压电信号 — ADC{self.piezo_adc_group + 1} {ch_name}")
            self.piezo_status_label.setText(f'已连接 {port}')
            self.status_bar.showMessage(f"压电传感器已连接: {port}，记录全部8通道")
        else:
            self.piezo_thread.stop()
            self.piezo_thread = None
            self.btn_piezo_connect.setText("连接压电")
            self.btn_piezo_connect.setStyleSheet("")
            self.piezo_status_label.setText('未连接')
            self.piezo_curve_preview.setData([])
            self.status_bar.showMessage("压电传感器已断开")

    def _refresh_piezo_plot(self):
        """由低频 plot_ready 信号触发，刷新波形（约5-10Hz，不影响视觉帧率）"""
        if self.piezo_thread is None:
            return
        y = self.piezo_thread.get_plot_data()
        if len(y) == 0:
            return
        self.piezo_curve_preview.setData(y)
        # 更新状态标签：显示缓冲区大小
        with self.piezo_thread._buf_lock:
            n = len(self.piezo_thread._group_bufs[self.piezo_thread._selected_adc_group])
        self.piezo_status_label.setText(
            f'ADC{self.piezo_adc_group + 1} CH{self.piezo_preview_ch + 1}  |  缓冲: {n} 帧')

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
        # 取消时不再自动关闭程序

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

        # 从matched_points.npz加载mirror_axis和image_shape
        mirror_axis = 640
        ref_image_shape = None  # 生成首帧数据时的图像尺寸 (h, w)
        history_file = os.path.join(self.data_save_dir, "matched_points.npz")
        if os.path.exists(history_file):
            try:
                with np.load(history_file) as npz_data:
                    if 'mirror_axis' in npz_data:
                        mirror_axis = int(npz_data['mirror_axis'])
                    if 'image_shape' in npz_data:
                        ref_image_shape = tuple(int(v) for v in npz_data['image_shape'])
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

        # 使用首帧数据生成时的图像尺寸进行立体校正，确保与 frame_000_points.txt 中的 3D 坐标一致
        if ref_image_shape is not None:
            h, w = ref_image_shape
        else:
            h, w = 480, 1280
        self.FRAME_DATA['ref_image_shape'] = (h, w)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T,
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

        pts1 = cv2.undistortPoints(pts1_R, K1, D1, R=R1, P=P1).squeeze()
        pts2 = cv2.undistortPoints(pts2_R, K2, D2, R=R2, P=P2).squeeze()

        # 更新全局数据
        self.FRAME_DATA.update({
            'initialized': True,
            'roi_masks': {'left': self.drawing['left']['mask'], 'right': self.drawing['right']['mask']},
            'P1': P1, 'P2': P2, 'R1': R1, 'R2': R2,
            'left_points_0': pts1, 'right_points_0': pts2,
            'left_points_0_R': pts1_R, 'right_points_0_R': pts2_R,
            'left_points_0_pre': pts1_R.copy(), 'right_points_0_pre': pts2_R.copy(),
            'base_3d_points': points_3d,
            'mirror_axis': mirror_axis,
        })

        # 构建坐标系
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix
            self.pca_flip_xy = self._detect_pca_orientation()

        # 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        return True

    def start_camera(self):
        """启动摄像头"""
        self.camera_index = self.spin_camera_idx.value()
        self.rotate_180 = self.chk_rotate.isChecked()

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
        self.camera_thread.fps_signal.connect(lambda fps: self._fps_label.setText(f"FPS: {fps:.1f}"))
        self.camera_thread.start()

        # 启用控制按钮
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("暂停")
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.is_playing = True

        self.status_bar.showMessage(f"摄像头已启动 (ID: {self.camera_index})")

    def on_camera_frame(self, frame, timestamp=0.0):
        """处理摄像头帧 - 仅分发到处理线程，不直接显示"""
        if not self.FRAME_DATA['initialized']:
            return

        self.latest_camera_frame = frame
        self.current_frame_idx += 1

        # 校验：实时帧尺寸必须与首帧数据生成时的尺寸一致，否则 P1/P2 会与 base_3d_points 不匹配
        h, w = frame.shape[:2]
        ref_shape = self.FRAME_DATA.get('ref_image_shape')
        if ref_shape is not None and (h, w) != ref_shape:
            if self.current_frame_idx == 1:
                QMessageBox.warning(self, "警告",
                    f"摄像头分辨率 {w}x{h} 与首帧数据生成时的分辨率 "
                    f"{ref_shape[1]}x{ref_shape[0]} 不一致，三维重建会出现偏移。")

        # 每10帧更新一次帧计数显示，减少GUI更新频率
        if self.current_frame_idx % 10 == 0:
            self.lbl_frame_info.setText(f"帧: {self.current_frame_idx}")

        if self.is_playing and self.process_thread is not None:
            # 将帧发送到处理线程（异步处理）
            self.process_thread.add_frame(frame.copy(), timestamp)
        else:
            # 暂停时仅显示原始帧
            self.display_frame(frame)

    def on_process_result(self, frame, points_3d, left_pts, right_pts, left_lost, right_lost, timestamp=0.0, is_abnormal=False):
        """处理线程返回结果 - 更新显示"""
        if points_3d is not None:
            # 采集缓冲
            self.buffer_frame_data(points_3d, timestamp, is_abnormal)

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
            'P1': P1, 'P2': P2, 'R1': R1, 'R2': R2,
            'left_points_0': pts1, 'right_points_0': pts2,
            'left_points_0_R': pts1_R, 'right_points_0_R': pts2_R,
            'left_points_0_pre': pts1_R.copy(), 'right_points_0_pre': pts2_R.copy(),
            'base_3d_points': points_3d,
            'mirror_axis': mirror_axis,
            'ref_image_shape': (h, w),
        })

        # 构建坐标系
        if len(points_3d) >= 3:
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix
            self.pca_flip_xy = self._detect_pca_orientation()

        # 提取基准帧Delaunay拓扑边
        self.FRAME_DATA['left_edges'] = self.extract_edges(pts1_R.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(pts2_R.astype(np.float64))

        # 显示首帧（重置scatter状态，确保坐标轴以局部坐标系重建）
        self.current_frame_idx = 0
        self._scatter_plot = None
        self._colorbar = None
        self._quiver_plot = None
        self.fig_3d.clear()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.display_frame(frame, pts1_R, pts2_R)
        self.update_3d_view(points_3d)

        # 启用控制按钮
        self.btn_play_pause.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_set_as_base.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.status_bar.showMessage(f"首帧处理完成，加载了{len(points_3d)}个标记点")

    def reset_first_frame(self):
        """重置到首帧"""
        self.current_frame_idx = 0

        self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
        self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()

        self._scatter_plot = None
        self._colorbar = None
        self._quiver_plot = None

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

        # 重新计算基准3D点 — 复用 load 时缓存的 R1/R2/P1/P2
        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R1 = self.FRAME_DATA['R1']
        R2 = self.FRAME_DATA['R2']
        P1 = self.FRAME_DATA['P1']
        P2 = self.FRAME_DATA['P2']

        l_pts_ud = cv2.undistortPoints(left_pts, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_pts, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

        self.FRAME_DATA['base_3d_points'] = points_3d

        # 重新计算 PCA 坐标系，传入上一次的旋转矩阵约束主轴符号，防止随机翻转
        if len(points_3d) >= 3:
            prev_rot = self.FRAME_DATA.get('transform_rotation')
            origin, rotation_matrix = self.build_coordinate_system_pca(points_3d, prev_rotation=prev_rot)
            self.FRAME_DATA['transform_origin'] = origin
            self.FRAME_DATA['transform_rotation'] = rotation_matrix

        # 基准帧拓扑边也要重建
        self.FRAME_DATA['left_edges'] = self.extract_edges(left_pts.astype(np.float64))
        self.FRAME_DATA['right_edges'] = self.extract_edges(right_pts.astype(np.float64))

        # 重置 scatter 对象，基准帧变化后需要重建
        self._scatter_plot = None
        self._colorbar = None
        self._quiver_plot = None
        self.fig_3d.clear()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d.draw_idle()

        # 更新3D视图
        self.update_3d_view(points_3d)

        self.status_bar.showMessage(f"已将当前帧设置为新的基准帧（帧 {self.current_frame_idx}）")

    # ---- HDF5 数据采集 ----
    def toggle_recording(self):
        """切换采集状态"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """开始数据采集"""
        if not self.FRAME_DATA['initialized']:
            self.status_bar.showMessage("请先初始化首帧")
            return

        self.recording_buffer = {'timestamps': [], 'xyz': [], 'abnormal': []}
        self.ft_recording_buffer = {'timestamps': [], 'ft_values': []}

        # 提前生成文件名，供流式录制和力觉文件使用（两者使用相同时间戳）
        start_time = datetime.now()
        self._current_record_start_time = start_time
        filename = f"calibration_{start_time.strftime('%Y%m%d_%H%M%S')}.h5"
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_calibration")
        os.makedirs(save_dir, exist_ok=True)
        self._current_record_filepath = os.path.join(save_dir, filename)
        self._current_record_filename = filename

        # 启动压电流式录制（直接写HDF5，防止长时间采集数据丢失）
        if self.piezo_thread is not None and self.piezo_thread.isRunning():
            self.piezo_thread.start_streaming_record(self._current_record_filepath)
        self._piezo_record_start_ts = time.time()

        self.is_recording = True
        self.btn_record.setText("停止采集")
        self.btn_record.setStyleSheet("background-color: #ff4444; color: white;")
        has_piezo = self.piezo_thread is not None and self.piezo_thread.isRunning()
        self.status_bar.showMessage(
            f"数据采集中... {'(含压电8通道，流式写入)' if has_piezo else '(无压电，可连接后再采集)'}")

    def stop_recording(self):
        """停止数据采集并保存HDF5"""
        self.is_recording = False
        self.btn_record.setText("开始采集")
        self.btn_record.setStyleSheet("")

        # 停止压电流式录制（刷新剩余缓冲并关闭文件句柄）
        if self.piezo_thread is not None:
            self.piezo_thread.stop_streaming_record()

        stop_time = datetime.now()
        self.save_to_hdf5(stop_time)
        self.save_ft_to_hdf5(stop_time)

    def get_local_coords(self, points_3d):
        """将3D点转换为局部坐标"""
        if self.FRAME_DATA['transform_rotation'] is not None:
            return self.transform_to_local_coordinates(points_3d)
        return points_3d.copy()

    def buffer_frame_data(self, points_3d, timestamp, is_abnormal=False):
        """将当前帧数据写入采集缓冲区"""
        if not self.is_recording or points_3d is None:
            return
        local_xyz = self.get_local_coords(points_3d)
        self.recording_buffer['timestamps'].append(timestamp)
        self.recording_buffer['xyz'].append(local_xyz.astype(np.float32))
        self.recording_buffer['abnormal'].append(is_abnormal)

    def save_to_hdf5(self, stop_time):
        """将视觉缓冲数据追加保存到流式录制的HDF5文件（压电数据已由流式录制写入）"""
        if not self.recording_buffer['timestamps']:
            QMessageBox.warning(self, "警告", "没有采集到数据")
            self.status_bar.showMessage("采集已停止（无数据）")
            return

        timestamps = np.array(self.recording_buffer['timestamps'], dtype=np.float64)
        xyz_all = np.array(self.recording_buffer['xyz'], dtype=np.float32)  # (T, N, 3)
        abnormal = np.array(self.recording_buffer['abnormal'], dtype=np.bool_)  # (T,)

        # xyz_ref 绑定当前 base_3d_points
        base_3d = self.FRAME_DATA['base_3d_points']
        xyz_ref = self.get_local_coords(base_3d).astype(np.float32)  # (N, 3)

        # 计算 dxyz
        dxyz = (xyz_all - xyz_ref[np.newaxis, :, :]).astype(np.float32)  # (T, N, 3)

        N = xyz_ref.shape[0]
        point_id = np.arange(N, dtype=np.int32)

        # 使用流式录制时已创建的文件（追加写入视觉数据）
        filepath = getattr(self, '_current_record_filepath', None)
        filename = getattr(self, '_current_record_filename', None)
        if filepath is None:
            # 兜底：若未启动流式录制（无压电），新建文件
            filename = f"calibration_{stop_time.strftime('%Y%m%d_%H%M%S')}.h5"
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_calibration")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            file_mode = 'w'
        else:
            file_mode = 'a'  # 追加到已有文件（压电流式数据已在其中）

        # 统计压电帧数（从已写入的流式文件中读取）
        piezo_sample_count = 0
        if file_mode == 'a':
            try:
                with h5py.File(filepath, 'r') as f_r:
                    if 'piezo_stream' in f_r:
                        for g in range(5):
                            grp_key = f'piezo_stream/adc{g+1}/timestamp'
                            if grp_key in f_r:
                                piezo_sample_count += f_r[grp_key].shape[0]
            except Exception:
                pass

        try:
            with h5py.File(filepath, file_mode) as f:
                # /vision
                vision = f.require_group('vision')
                vision.create_dataset('timestamp', data=timestamps)
                vision.create_dataset('xyz', data=xyz_all)
                vision.create_dataset('dxyz', data=dxyz)
                vision.create_dataset('point_id', data=point_id)
                vision.create_dataset('abnormal', data=abnormal)

                # /reference
                ref = f.require_group('reference')
                if 'xyz_ref' not in ref:
                    ref.create_dataset('xyz_ref', data=xyz_ref)

                # /meta
                meta = f.require_group('meta')
                meta.attrs['camera_fps'] = self.spin_speed.value()
                meta.attrs['marker_count'] = N
                meta.attrs['experiment_name'] = filename
                meta.attrs['has_piezo'] = (piezo_sample_count > 0)
                meta.attrs['pca_flip_xy'] = int(self.pca_flip_xy)

                # /force（预留扩展位置）
                f.require_group('force')

            self.recording_buffer = {'timestamps': [], 'xyz': [], 'abnormal': []}

            piezo_info = f"\n压电采样: {piezo_sample_count} 点 × 8通道（流式写入，全部ADC组）" if piezo_sample_count > 0 else ""
            QMessageBox.information(self, "保存成功",
                f"数据已保存到:\n{filepath}\n\n帧数: {len(timestamps)}\nMarker数: {N}{piezo_info}")
            self.status_bar.showMessage(f"采集完成，已保存 {len(timestamps)} 帧到 {filename}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入HDF5文件失败:\n{e}")
            self.status_bar.showMessage(f"保存失败: {e}")

    def save_ft_to_hdf5(self, stop_time):
        """将力传感器缓冲数据保存为单独的HDF5文件"""
        with self.ft_lock:
            ft_timestamps = self.ft_recording_buffer['timestamps'].copy()
            ft_values = self.ft_recording_buffer['ft_values'].copy()

        if not ft_timestamps:
            print("力传感器无采集数据，跳过保存")
            return

        timestamps = np.array(ft_timestamps, dtype=np.float64)
        values = np.array(ft_values, dtype=np.float64)  # (T, 6)

        start_time = getattr(self, '_current_record_start_time', stop_time)
        filename = f"ft_calibration_{start_time.strftime('%Y%m%d_%H%M%S')}.h5"
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_calibration")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        try:
            with h5py.File(filepath, 'w') as f:
                # /force
                force = f.create_group('force')
                force.create_dataset('timestamp', data=timestamps)
                force.create_dataset('values', data=values)  # (T, 6): fx,fy,fz,mx,my,mz
                force.attrs['columns'] = 'fx,fy,fz,mx,my,mz'

                # /meta
                meta = f.create_group('meta')
                meta.attrs['experiment_name'] = filename
                meta.attrs['sample_count'] = len(timestamps)

            with self.ft_lock:
                self.ft_recording_buffer = {'timestamps': [], 'ft_values': []}

            QMessageBox.information(self, "力传感器数据保存成功",
                f"数据已保存到:\n{filepath}\n\n采样数: {len(timestamps)}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入力传感器HDF5文件失败:\n{e}")

    def process_frame(self, frame):
        """处理后续帧 - 带mesh untangling和灾难性崩溃检测"""
        if not self.FRAME_DATA['initialized']:
            return None, None, None, None, None, False

        h, w = frame.shape[:2]
        mirror_axis = self.FRAME_DATA['mirror_axis']

        # 生成镜像视图
        if self._left_img_buf is None or self._left_img_buf.shape != frame.shape:
            self._left_img_buf = np.full((h, w, 3), 255, dtype=np.uint8)
            self._right_img_buf = np.full((h, w, 3), 255, dtype=np.uint8)
        self._left_img_buf[:] = 255
        self._left_img_buf[:, :mirror_axis] = frame[:, :mirror_axis]
        self._right_img_buf[:] = 255
        self._right_img_buf[:, mirror_axis:] = frame[:, mirror_axis:]
        left_img = self._left_img_buf
        right_img = self._right_img_buf

        # 检测标记点
        left_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                  self.apply_roi_mask(self.detector.detect(left_img),
                                                      self.drawing['left']['mask'])], dtype=np.float32)
        right_pts_det_raw = np.array([(x, y) for (x, y, _) in
                                   self.apply_roi_mask(self.detector.detect(right_img),
                                                       self.drawing['right']['mask'])], dtype=np.float32)

        # 过滤掉距离所有参考点都太远的检测点
        pre_l = self.FRAME_DATA['left_points_0_pre']
        pre_r = self.FRAME_DATA['right_points_0_pre']
        filter_dist = self.max_match_dist * 1.5

        if id(pre_l) != self._pre_l_tree_id:
            self._pre_l_tree = cKDTree(pre_l) if len(pre_l) > 0 else None
            self._pre_l_tree_id = id(pre_l)
        if id(pre_r) != self._pre_r_tree_id:
            self._pre_r_tree = cKDTree(pre_r) if len(pre_r) > 0 else None
            self._pre_r_tree_id = id(pre_r)

        if len(left_pts_det_raw) > 0 and self._pre_l_tree is not None:
            min_dists_l, _ = self._pre_l_tree.query(left_pts_det_raw, k=1)
            valid_mask_l = min_dists_l <= filter_dist
            left_pts_det = left_pts_det_raw[valid_mask_l]
        else:
            left_pts_det = left_pts_det_raw

        if len(right_pts_det_raw) > 0 and self._pre_r_tree is not None:
            min_dists_r, _ = self._pre_r_tree.query(right_pts_det_raw, k=1)
            valid_mask_r = min_dists_r <= filter_dist
            right_pts_det = right_pts_det_raw[valid_mask_r]
        else:
            right_pts_det = right_pts_det_raw

        # 匹配
        matched_pairs = self.auto_match_points(
            left_pts_det, right_pts_det,
            self.FRAME_DATA['left_points_0_pre'],
            self.FRAME_DATA['right_points_0_pre'],
            max_dist=self.max_match_dist
        )

        # 局部近邻运动补偿
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

        # ---- Mesh untangling：基于拓扑边检测交叉并修复 ----
        left_edges = self.FRAME_DATA.get('left_edges', [])
        right_edges = self.FRAME_DATA.get('right_edges', [])

        def count_crossings(pts, edges):
            crossings = []
            for i in range(len(edges)):
                a1, b1 = edges[i]
                for j in range(i + 1, len(edges)):
                    a2, b2 = edges[j]
                    if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
                        continue
                    if self.segments_intersect(pts[a1], pts[b1], pts[a2], pts[b2]):
                        crossings.append((i, j, a1, b1, a2, b2))
            return crossings

        def untangle(pts, edges, max_iter=5):
            pts = pts.copy()
            for _ in range(max_iter):
                crossings = count_crossings(pts, edges)
                if not crossings:
                    return pts, []
                for _, _, a1, b1, a2, b2 in crossings:
                    pts[b1], pts[b2] = pts[b2].copy(), pts[b1].copy()
                crossings = count_crossings(pts, edges)
                if not crossings:
                    return pts, []
            return pts, crossings

        left_crossings = []
        right_crossings = []

        if left_edges:
            left_points_R, left_crossings = untangle(left_points_R, left_edges)
        if right_edges:
            right_points_R, right_crossings = untangle(right_points_R, right_edges)

        self.FRAME_DATA['current_left_crossings'] = left_crossings
        self.FRAME_DATA['current_right_crossings'] = right_crossings

        # ---- 灾难性崩溃检测 ----
        n_total = len(pre_l)
        lost_ratio = len(lost_indices) / max(n_total, 1)
        total_crossings = len(left_crossings) + len(right_crossings)
        is_catastrophic = (lost_ratio > 0.5 or total_crossings > 20)

        if is_catastrophic:
            self.consecutive_abnormal_frames += 1
        else:
            self.consecutive_abnormal_frames = 0

        # ---- 分级异常帧处理 ----
        if self.consecutive_abnormal_frames > 0 and self.consecutive_abnormal_frames < 5:
            # 时间冻结：保持上一帧数据不变，不更新参考点
            return (self.FRAME_DATA['base_3d_points'].copy(),
                    self.FRAME_DATA['left_points_0_R'].copy(),
                    self.FRAME_DATA['right_points_0_R'].copy(),
                    np.zeros(n_total, dtype=bool),
                    np.zeros(n_total, dtype=bool),
                    True)

        if self.consecutive_abnormal_frames >= 5:
            # 强制重置：连续异常超过5帧，重置参考点到首帧
            self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
            self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
            self.consecutive_abnormal_frames = 0
            return (self.FRAME_DATA['base_3d_points'].copy(),
                    self.FRAME_DATA['left_points_0_R'].copy(),
                    self.FRAME_DATA['right_points_0_R'].copy(),
                    np.zeros(n_total, dtype=bool),
                    np.zeros(n_total, dtype=bool),
                    True)

        # 三维重建 — 复用首帧 load 时算好的 R1/R2/P1/P2，确保和 base_3d_points 同一坐标系
        left_points_R = left_points_R.astype(np.float32)
        right_points_R = right_points_R.astype(np.float32)

        K1, D1 = self.stereo_params['K1'], self.stereo_params['D1']
        K2, D2 = self.stereo_params['K2'], self.stereo_params['D2']
        R1 = self.FRAME_DATA['R1']
        R2 = self.FRAME_DATA['R2']
        P1 = self.FRAME_DATA['P1']
        P2 = self.FRAME_DATA['P2']

        l_pts_ud = cv2.undistortPoints(left_points_R, K1, D1, R=R1, P=P1).squeeze()
        r_pts_ud = cv2.undistortPoints(right_points_R, K2, D2, R=R2, P=P2).squeeze()
        points_3d = self.linear_triangulation(l_pts_ud, r_pts_ud, P1, P2)

        # 异常帧检测（3D层面）
        base_3d = self.FRAME_DATA['base_3d_points']
        is_abnormal = False

        if base_3d is not None and len(points_3d) == len(base_3d):
            z_diff = points_3d[:, 2] - base_3d[:, 2]
            if np.any(z_diff > 1.0):
                is_abnormal = True

            if not is_abnormal and len(points_3d) >= 2:
                if len(cKDTree(points_3d).query_pairs(0.3)) > 0:
                    is_abnormal = True

            if is_abnormal:
                self.FRAME_DATA['left_points_0_pre'] = self.FRAME_DATA['left_points_0_R'].copy()
                self.FRAME_DATA['right_points_0_pre'] = self.FRAME_DATA['right_points_0_R'].copy()
                return (self.FRAME_DATA['base_3d_points'].copy(),
                        self.FRAME_DATA['left_points_0_R'].copy(),
                        self.FRAME_DATA['right_points_0_R'].copy(),
                        np.zeros(len(base_3d), dtype=bool),
                        np.zeros(len(base_3d), dtype=bool),
                        True)

        # 更新参考点
        self.FRAME_DATA['left_points_0_pre'] = left_points_R
        self.FRAME_DATA['right_points_0_pre'] = right_points_R

        return points_3d, left_points_R, right_points_R, left_lost_mask, right_lost_mask, False

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
        pts1_h = pts1.T.reshape(2, -1).astype(np.float64)
        pts2_h = pts2.T.reshape(2, -1).astype(np.float64)
        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        pts4d /= pts4d[3]
        return pts4d[:3].T

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

    def build_coordinate_system_pca(self, points_3d, prev_rotation=None):
        """使用PCA构建坐标系。
        prev_rotation: 上一次的旋转矩阵（3x3），用于约束主轴符号，防止随机翻转。
        首次调用传 None，后续调用传已有的旋转矩阵。
        """
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
            # 用前一次旋转矩阵的对应列约束符号：点积 < 0 说明方向相反，翻转
            prev_x = prev_rotation[:, 0]
            prev_z = prev_rotation[:, 2]
            if np.dot(z_axis, prev_z) < 0:
                z_axis = -z_axis
            if np.dot(x_axis, prev_x) < 0:
                x_axis = -x_axis
        else:
            # 首次初始化：用世界坐标系 Z 轴方向约束（传感器法线大致朝 +Z）
            if z_axis[2] < 0:
                z_axis = -z_axis
            # X 轴用世界坐标系 X 分量约束
            if x_axis[0] < 0:
                x_axis = -x_axis

        # 重新正交化，保证右手系
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        return centroid, rotation_matrix

    def _detect_pca_orientation(self):
        """自动检测点云是否需要绕Z轴旋转180°。
        规则点阵（3行）在正确取向下，最上一行Y > 6mm。
        如果没有点满足Y > 6，说明Y轴方向反了，需要翻转XY。
        """
        origin = self.FRAME_DATA.get('transform_origin')
        rotation = self.FRAME_DATA.get('transform_rotation')
        points_3d = self.FRAME_DATA.get('base_3d_points')
        if origin is None or rotation is None or points_3d is None:
            return False
        local = np.dot(points_3d - origin, rotation)
        return np.sum(local[:, 1] > 6) == 0

    def transform_to_local_coordinates(self, points):
        """转换到局部坐标系"""
        origin = self.FRAME_DATA['transform_origin']
        rotation = self.FRAME_DATA['transform_rotation']
        translated = points - origin
        result = np.dot(translated, rotation)
        if self.pca_flip_xy:
            result[:, 0] = -result[:, 0]
            result[:, 1] = -result[:, 1]
        return result

    def display_frame(self, frame, left_pts=None, right_pts=None, left_lost=None, right_lost=None):
        """显示帧"""
        display_img = frame.copy()

        # 绘制Delaunay mesh边（在标记点之前绘制，这样点在边之上）
        left_edges = self.FRAME_DATA.get('left_edges', [])
        right_edges = self.FRAME_DATA.get('right_edges', [])
        left_crossings = self.FRAME_DATA.get('current_left_crossings', [])
        right_crossings = self.FRAME_DATA.get('current_right_crossings', [])

        # 收集交叉边索引
        left_cross_set = set()
        for item in left_crossings:
            left_cross_set.add(item[0])
            left_cross_set.add(item[1])
        right_cross_set = set()
        for item in right_crossings:
            right_cross_set.add(item[0])
            right_cross_set.add(item[1])

        if left_pts is not None and left_edges:
            for ei, (a, b) in enumerate(left_edges):
                if a < len(left_pts) and b < len(left_pts):
                    p1 = (int(left_pts[a][0]), int(left_pts[a][1]))
                    p2 = (int(left_pts[b][0]), int(left_pts[b][1]))
                    color = (0, 0, 255) if ei in left_cross_set else (180, 180, 180)
                    cv2.line(display_img, p1, p2, color, 1)

        if right_pts is not None and right_edges:
            for ei, (a, b) in enumerate(right_edges):
                if a < len(right_pts) and b < len(right_pts):
                    p1 = (int(right_pts[a][0]), int(right_pts[a][1]))
                    p2 = (int(right_pts[b][0]), int(right_pts[b][1]))
                    color = (0, 0, 255) if ei in right_cross_set else (180, 180, 180)
                    cv2.line(display_img, p1, p2, color, 1)

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
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

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
        if self.is_dragging:
            return
        if time.time() - self.drag_release_time < self.drag_cooldown:
            return

        if not hasattr(self, '_scatter_plot'):
            self._scatter_plot = None
            self._colorbar = None
            self._quiver_plot = None

        if self.FRAME_DATA['transform_rotation'] is not None:
            points = self.transform_to_local_coordinates(points_3d)

            base_local = self.transform_to_local_coordinates(self.FRAME_DATA['base_3d_points'])

            displacement_vectors = points - base_local
            deformation = np.linalg.norm(displacement_vectors, axis=1)

            self.calculate_displacement_stats(displacement_vectors, deformation)
            self.current_points_3d = points

            if self._scatter_plot is None:
                self.fig_3d.clear()
                ax = self.fig_3d.add_subplot(111, projection='3d')
                self.ax_3d = ax
                self._scatter_plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                                c=deformation, cmap='jet', s=50, vmin=0, vmax=1.5)
                self._colorbar = self.fig_3d.colorbar(self._scatter_plot, ax=ax, shrink=0.8)
                self._colorbar.set_label('Deformation (mm)', rotation=270, labelpad=15)
                self._quiver_plot = None
            else:
                ax = self.ax_3d
                self._scatter_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                self._scatter_plot.set_array(deformation)

            if self._quiver_plot is not None:
                try:
                    self._quiver_plot.remove()
                except (ValueError, AttributeError):
                    pass
                self._quiver_plot = None

            if self.avg_displacement_magnitude > 0.01:
                center = np.mean(points, axis=0)
                arrow_scale = 5.0
                self._quiver_plot = ax.quiver(
                    center[0], center[1], center[2],
                    self.avg_displacement_vector[0] * arrow_scale,
                    self.avg_displacement_vector[1] * arrow_scale,
                    self.avg_displacement_vector[2] * arrow_scale,
                    color='red', arrow_length_ratio=0.3, linewidth=2.5,
                    label=f'Avg: {self.avg_displacement_magnitude:.3f}mm')
                ax.legend(loc='upper left', fontsize=8)
        else:
            points = points_3d.copy()
            points[:, 1] = -points[:, 1]

            if self._scatter_plot is None:
                self.fig_3d.clear()
                ax = self.fig_3d.add_subplot(111, projection='3d')
                self.ax_3d = ax
                self._scatter_plot = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=50)
            else:
                ax = self.ax_3d
                self._scatter_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

            self.reset_displacement_stats()

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
                ax.view_init(elev=90, azim=180, roll=-90)
            except TypeError:
                ax.view_init(elev=90, azim=180)

        self.canvas_3d.draw_idle()

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
        frame_timestamp = time.time()
        if not ret:
            self.pause_playback()
            return

        if self.current_frame_idx == 0:
            # 首帧直接显示
            self.display_frame(frame, self.FRAME_DATA['left_points_0_R'], self.FRAME_DATA['right_points_0_R'])
            self.update_3d_view(self.FRAME_DATA['base_3d_points'])
            self.buffer_frame_data(self.FRAME_DATA['base_3d_points'], frame_timestamp, False)
        else:
            # 处理后续帧
            points_3d, left_pts, right_pts, left_lost, right_lost, is_abnormal = self.process_frame(frame)
            if points_3d is not None:
                self.display_frame(frame, left_pts, right_pts, left_lost, right_lost)
                self.update_3d_view(points_3d)
                self.buffer_frame_data(points_3d, frame_timestamp, is_abnormal)

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
                points_3d, left_pts, right_pts, left_lost, right_lost, _abnormal = self.process_frame(frame)

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

    def closeEvent(self, event):
        """关闭窗口"""
        # 如果正在采集，先停止并保存
        if self.is_recording:
            self.stop_recording()
        # 停止力传感器显示刷新
        self.ft_display_timer.stop()
        # 停止播放定时器
        self.play_timer.stop()
        # 停止压电传感器线程
        if self.piezo_thread is not None:
            self.piezo_thread.stop()
            self.piezo_thread = None
        # 关闭力传感器订阅
        if self.ft_node is not None:
            # 先清除订阅引用，阻止回调继续触发
            self.ft_subscription = None
            try:
                self.ft_node.Shutdown()
            except Exception:
                pass
            self.ft_node = None
        # 先设置 running=False 让线程自行退出，再 wait
        if self.process_thread is not None:
            self.process_thread.running = False
            self.process_thread.has_new_frame.set()
        if self.camera_thread is not None:
            self.camera_thread.running = False
        # 释放视频资源（这样 camera_thread 的 read() 能更快返回）
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        # 等待线程结束
        if self.process_thread is not None:
            self.process_thread.stop()
            self.process_thread = None
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 让 Ctrl+C 能正常关闭程序
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    # 需要一个定时器让 Python 有机会处理信号
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    window = VideoPointCloudPlayer()
    window.show()
    ret = app.exec_()

    # 强制退出，确保 ZMQ/订阅残留线程不会阻止进程结束
    os._exit(ret)
