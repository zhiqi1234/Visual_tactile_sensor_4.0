# -*- coding: utf-8 -*-
"""HDF5 标定数据编辑器 — 时间戳对齐 / 有效段筛选 / 保存处理数据"""
import sys
import os
import numpy as np
import h5py
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit,
    QGroupBox, QSlider, QSpinBox, QDoubleSpinBox,
    QDockWidget, QListWidget, QAbstractItemView, QMessageBox,
    QScrollBar, QSplitter, QComboBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 标定数据编辑器")
        self.setGeometry(100, 100, 1400, 900)

        self.h5_data = {}
        self.ft_data = {}
        self.piezo_data = {}  # 压电数据
        self.total_frames = 0
        self._saved_view = None
        self._saved_force3d_view = None

        # 时间戳对齐
        self.time_offset = 0.0

        # 有效数据段
        self.segments = []
        self._dragging_segment = None
        self._seg_artists = []  # [(span_f, span_m, vline_s_f, vline_s_m, vline_e_f, vline_e_m), ...]
        self._force_axes = None  # (ax_f, ax_m) 缓存
        self._force_plot_cached = False  # 力觉图表是否已缓存
        self._force_frame_markers = []  # 当前帧标记线引用，用于清理

        # 源文件路径
        self._source_pc_path = ""
        self._source_ft_path = ""

        # 再编辑标志（处理后文件加载时设为True）
        self._is_from_processed = False

        # 力传感器静态偏差
        self.force_bias = np.zeros(6)

        # 压电传感器相关
        self.piezo_channel = 0  # 当前选择的通道（0-7）
        self.piezo_adc_group = 1  # 当前选择的ADC组（0-4，默认ADC2）
        self.piezo_window_ms = 33  # 时间窗口（毫秒）

        # 文件导航
        self._current_dir = ""
        self._file_list = []
        self._current_file_index = -1

        self._apply_stylesheet()
        self.init_ui()
        self.statusBar().showMessage("就绪 — 请打开 HDF5 文件")

    # ──────────────────────────────────────────────
    #  样式表
    # ──────────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F5F6FA; }
            QGroupBox {
                background-color: #FFFFFF; border: 1px solid #D0D5DD;
                border-radius: 6px; margin-top: 14px; padding-top: 10px;
                font-weight: bold; font-size: 13px; color: #344054;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 2px 10px; background-color: #FFFFFF;
                border: 1px solid #D0D5DD; border-radius: 4px; left: 10px;
            }
            QPushButton#btnOpen, QPushButton#btnOpenForce,
            QPushButton#btnOpenProcessed, QPushButton#btnSave {
                background-color: #4A90D9; color: white; border: none;
                border-radius: 5px; padding: 7px 16px;
                font-size: 13px; font-weight: bold; min-width: 120px;
            }
            QPushButton#btnOpen:hover, QPushButton#btnOpenForce:hover,
            QPushButton#btnOpenProcessed:hover, QPushButton#btnSave:hover {
                background-color: #357ABD;
            }
            QPushButton#btnSave { background-color: #27AE60; }
            QPushButton#btnSave:hover { background-color: #219A52; }
            QPushButton#btnToggleInfo {
                background-color: #667085; color: white; border: none;
                border-radius: 5px; padding: 7px 14px;
                font-size: 12px; min-width: 80px;
            }
            QPushButton#btnToggleInfo:checked { background-color: #4A90D9; }
            QPushButton#btnAddSeg, QPushButton#btnDelSeg {
                background-color: #E4E7EC; color: #344054; border: none;
                border-radius: 4px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton#btnAddSeg:hover { background-color: #D0D5DD; }
            QPushButton#btnDelSeg:hover { background-color: #FDA29B; color: white; }
            QLabel#lblFile, QLabel#lblForceFile {
                background-color: #FFFFFF; border: 1px solid #D0D5DD;
                border-radius: 4px; padding: 6px 10px;
                color: #667085; font-size: 12px;
            }
            QTextEdit {
                background-color: #FFFFFF; border: 1px solid #D0D5DD;
                border-radius: 4px; font-family: Consolas, 'Courier New', monospace;
                font-size: 12px; padding: 6px; color: #1D2939;
            }
            QSlider::groove:horizontal {
                border: none; height: 6px; background: #E4E7EC; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4A90D9; border: 2px solid #FFFFFF;
                width: 16px; height: 16px; margin: -6px 0; border-radius: 9px;
            }
            QSlider::sub-page:horizontal { background: #4A90D9; border-radius: 3px; }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #D0D5DD; border-radius: 4px;
                padding: 4px 8px; font-size: 12px; min-height: 24px;
                background-color: #FFFFFF;
            }
            QLabel#lblFrame {
                font-size: 13px; font-weight: bold; color: #344054; min-width: 100px;
            }
            QStatusBar {
                background-color: #EAECF0; color: #475467;
                font-size: 12px; border-top: 1px solid #D0D5DD;
            }
            QDockWidget {
                font-size: 13px; font-weight: bold; color: #344054;
            }
            QDockWidget::title {
                background-color: #EAECF0; padding: 6px;
                border: 1px solid #D0D5DD;
            }
            QListWidget {
                background-color: #FFFFFF; border: 1px solid #D0D5DD;
                border-radius: 4px; font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #4A90D9; color: white;
            }
        """)

    # ──────────────────────────────────────────────
    #  UI 布局
    # ──────────────────────────────────────────────
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 6)
        main_layout.setSpacing(8)

        # ── 顶部按钮行 ──
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        self.btn_open = QPushButton("打开点云文件")
        self.btn_open.setObjectName("btnOpen")
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.clicked.connect(self.open_file)

        self.btn_prev_file = QPushButton("上一个文件")
        self.btn_prev_file.setObjectName("btnPrevFile")
        self.btn_prev_file.setCursor(Qt.PointingHandCursor)
        self.btn_prev_file.clicked.connect(self._load_prev_file)
        self.btn_prev_file.setEnabled(False)

        self.btn_next_file = QPushButton("下一个文件")
        self.btn_next_file.setObjectName("btnNextFile")
        self.btn_next_file.setCursor(Qt.PointingHandCursor)
        self.btn_next_file.clicked.connect(self._load_next_file)
        self.btn_next_file.setEnabled(False)

        self.btn_open_force = QPushButton("打开力觉文件")
        self.btn_open_force.setObjectName("btnOpenForce")
        self.btn_open_force.setCursor(Qt.PointingHandCursor)
        self.btn_open_force.clicked.connect(self.open_force_file)

        self.btn_open_processed = QPushButton("打开处理文件")
        self.btn_open_processed.setObjectName("btnOpenProcessed")
        self.btn_open_processed.setCursor(Qt.PointingHandCursor)
        self.btn_open_processed.clicked.connect(self._open_processed_file)

        self.btn_save = QPushButton("保存处理数据")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(self._save_processed)

        self.btn_toggle_info = QPushButton("信息面板")
        self.btn_toggle_info.setObjectName("btnToggleInfo")
        self.btn_toggle_info.setCheckable(True)
        self.btn_toggle_info.setChecked(False)
        self.btn_toggle_info.setCursor(Qt.PointingHandCursor)

        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.btn_prev_file)
        top_layout.addWidget(self.btn_next_file)
        top_layout.addWidget(self.btn_open_force)
        top_layout.addWidget(self.btn_open_processed)
        top_layout.addWidget(self.btn_save)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_toggle_info)
        main_layout.addLayout(top_layout)

        # ── 文件路径标签 ──
        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)
        self.lbl_file = QLabel("未选择点云文件")
        self.lbl_file.setObjectName("lblFile")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_force_file = QLabel("未选择力觉文件")
        self.lbl_force_file.setObjectName("lblForceFile")
        self.lbl_force_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_layout.addWidget(self.lbl_file, 1)
        path_layout.addWidget(self.lbl_force_file, 1)
        main_layout.addLayout(path_layout)

        # ── 时间偏移控件 ──
        offset_layout = QHBoxLayout()
        offset_layout.setSpacing(8)
        lbl_offset = QLabel("时间偏移:")
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(-10.0, 10.0)
        self.spin_offset.setSingleStep(0.001)
        self.spin_offset.setDecimals(3)
        self.spin_offset.setValue(0.0)
        self.spin_offset.setPrefix("Δt = ")
        self.spin_offset.setSuffix(" s")
        self.spin_offset.setToolTip("正值: 点云时间戳向后偏移; 负值: 向前偏移")
        self.spin_offset.valueChanged.connect(self._on_offset_changed)
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(lambda: self.spin_offset.setValue(0.0))
        offset_layout.addWidget(lbl_offset)
        offset_layout.addWidget(self.spin_offset)
        offset_layout.addWidget(btn_reset)
        offset_layout.addStretch()
        main_layout.addLayout(offset_layout)

        # ── 帧浏览 ──
        frame_group = QGroupBox("帧浏览")
        frame_inner = QHBoxLayout(frame_group)
        frame_inner.setContentsMargins(10, 20, 10, 8)
        frame_inner.setSpacing(12)
        self.lbl_frame = QLabel("帧: 0 / 0")
        self.lbl_frame.setObjectName("lblFrame")
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setPrefix("帧号: ")
        self.spin_frame.valueChanged.connect(self.on_frame_changed)
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self.spin_frame.setValue)
        self.spin_frame.valueChanged.connect(self.slider_frame.setValue)
        frame_inner.addWidget(self.lbl_frame)
        frame_inner.addWidget(self.spin_frame)
        frame_inner.addWidget(self.slider_frame, 1)
        main_layout.addWidget(frame_group)

        # ── 3D 可视化（点云 + 力向量 横排） ──
        plot_group = QGroupBox("3D 可视化")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(8, 20, 8, 8)

        plot_splitter = QSplitter(Qt.Horizontal)

        # 左侧：3D 点云
        pc_widget = QWidget()
        pc_layout = QVBoxLayout(pc_widget)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        pc_layout.addWidget(self.toolbar)
        pc_layout.addWidget(self.canvas)
        plot_splitter.addWidget(pc_widget)

        # 右侧：3D 力向量
        force_3d_widget = QWidget()
        force_3d_layout = QVBoxLayout(force_3d_widget)
        force_3d_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_force3d = plt.figure(figsize=(4, 4))
        self.canvas_force3d = FigureCanvas(self.fig_force3d)
        force_3d_layout.addWidget(QLabel("力/力矩 3D 向量"))
        force_3d_layout.addWidget(self.canvas_force3d)
        plot_splitter.addWidget(force_3d_widget)

        # 初始比例 (左 65%, 右 35%)
        plot_splitter.setStretchFactor(0, 65)
        plot_splitter.setStretchFactor(1, 35)

        plot_layout.addWidget(plot_splitter)
        main_layout.addWidget(plot_group, 3)

        # ── 压电信号控件 ──
        piezo_control_group = QGroupBox("压电信号")
        piezo_control_layout = QHBoxLayout(piezo_control_group)
        piezo_control_layout.setContentsMargins(8, 20, 8, 8)

        lbl_piezo_adc = QLabel("ADC组:")
        self.cbb_piezo_adc_group = QComboBox()
        self.cbb_piezo_adc_group.addItems([f"ADC{i+1}" for i in range(5)])
        self.cbb_piezo_adc_group.setCurrentIndex(1)  # 默认 ADC2
        self.cbb_piezo_adc_group.currentIndexChanged.connect(self._on_piezo_adc_group_changed)

        lbl_piezo_ch = QLabel("通道:")
        self.cbb_piezo_channel = QComboBox()
        self.cbb_piezo_channel.addItems([f"CH{i+1}" for i in range(8)])
        self.cbb_piezo_channel.setCurrentIndex(0)
        self.cbb_piezo_channel.currentIndexChanged.connect(self._on_piezo_channel_changed)

        lbl_piezo_window = QLabel("窗口:")
        self.spin_piezo_window = QSpinBox()
        self.spin_piezo_window.setRange(10, 100)
        self.spin_piezo_window.setValue(33)
        self.spin_piezo_window.setSuffix(" ms")
        self.spin_piezo_window.valueChanged.connect(self._on_piezo_window_changed)

        piezo_control_layout.addWidget(lbl_piezo_adc)
        piezo_control_layout.addWidget(self.cbb_piezo_adc_group)
        piezo_control_layout.addWidget(lbl_piezo_ch)
        piezo_control_layout.addWidget(self.cbb_piezo_channel)
        piezo_control_layout.addWidget(lbl_piezo_window)
        piezo_control_layout.addWidget(self.spin_piezo_window)
        piezo_control_layout.addStretch()
        main_layout.addWidget(piezo_control_group)

        # 压电波形图（Matplotlib）
        piezo_plot_group = QGroupBox("压电信号波形")
        piezo_plot_layout = QVBoxLayout(piezo_plot_group)
        piezo_plot_layout.setContentsMargins(8, 20, 8, 8)

        self.fig_piezo = plt.figure(figsize=(8, 2))
        self.canvas_piezo = FigureCanvas(self.fig_piezo)
        self.canvas_piezo.setMinimumHeight(150)
        self.toolbar_piezo = NavigationToolbar(self.canvas_piezo, self)
        self.piezo_ax = self.fig_piezo.add_subplot(111)
        self.piezo_ax.set_ylabel('电压 (V)')
        self.piezo_ax.set_xlabel('时间 (s)')
        self.piezo_ax.grid(True, alpha=0.3)
        (self.piezo_line,) = self.piezo_ax.plot([], [], 'b-', linewidth=0.8)
        self.piezo_marker = self.piezo_ax.axvline(0, color='r', linewidth=1.5,
                                                    linestyle='--', alpha=0.8)
        self.piezo_ax.set_title("压电信号 (无数据)", fontsize=10)
        self.fig_piezo.tight_layout()

        piezo_plot_layout.addWidget(self.toolbar_piezo)
        piezo_plot_layout.addWidget(self.canvas_piezo)
        main_layout.addWidget(piezo_plot_group, 1)

        # ── 力觉数据 + 有效段控件 ──
        force_plot_group = QGroupBox("力觉数据")
        force_plot_layout = QVBoxLayout(force_plot_group)
        force_plot_layout.setContentsMargins(8, 20, 8, 8)

        # 有效段控制栏
        seg_bar = QHBoxLayout()
        seg_bar.setSpacing(8)
        self.btn_add_segment = QPushButton("添加有效段")
        self.btn_add_segment.setObjectName("btnAddSeg")
        self.btn_add_segment.clicked.connect(self._add_segment)
        self.btn_del_segment = QPushButton("删除选中段")
        self.btn_del_segment.setObjectName("btnDelSeg")
        self.btn_del_segment.clicked.connect(self._del_segment)
        self.btn_auto_segment = QPushButton("自动选择有效段")
        self.btn_auto_segment.setObjectName("btnAddSeg")
        self.btn_auto_segment.clicked.connect(self._auto_detect_segments)
        self.btn_auto_press_segment = QPushButton("自动选择按压过程")
        self.btn_auto_press_segment.setObjectName("btnAddSeg")
        self.btn_auto_press_segment.clicked.connect(self._auto_detect_press_segments)
        self.list_segments = QListWidget()
        self.list_segments.setMaximumHeight(55)
        self.list_segments.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_segments.currentRowChanged.connect(self._on_segment_selected)
        seg_bar.addWidget(self.btn_add_segment)
        seg_bar.addWidget(self.btn_del_segment)
        seg_bar.addWidget(self.btn_auto_segment)
        seg_bar.addWidget(self.btn_auto_press_segment)
        seg_bar.addWidget(self.list_segments, 1)
        force_plot_layout.addLayout(seg_bar)

        # 力偏差控制栏
        bias_bar = QHBoxLayout()
        bias_bar.setSpacing(6)
        bias_bar.addWidget(QLabel("静态偏差:"))
        self.bias_spins = []
        bias_labels = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
        for name in bias_labels:
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 11px; color: #667085; min-width: 16px;")
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setValue(0.0)
            spin.setMaximumWidth(90)
            spin.valueChanged.connect(self._on_bias_changed)
            bias_bar.addWidget(lbl)
            bias_bar.addWidget(spin)
            self.bias_spins.append(spin)
        btn_auto_zero = QPushButton("自动归零")
        btn_auto_zero.setObjectName("btnAddSeg")
        btn_auto_zero.clicked.connect(self._auto_zero_bias)
        btn_reset_bias = QPushButton("重置偏差")
        btn_reset_bias.setObjectName("btnAddSeg")
        btn_reset_bias.clicked.connect(self._reset_bias)
        bias_bar.addWidget(btn_auto_zero)
        bias_bar.addWidget(btn_reset_bias)
        bias_bar.addStretch()
        force_plot_layout.addLayout(bias_bar)

        # 2D 力觉时序图
        force_2d_widget = QWidget()
        force_2d_layout = QVBoxLayout(force_2d_widget)
        force_2d_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_force = plt.figure(figsize=(6, 3))
        self.canvas_force = FigureCanvas(self.fig_force)
        self.canvas_force.setMinimumHeight(200)
        self.toolbar_force = NavigationToolbar(self.canvas_force, self)
        force_2d_layout.addWidget(self.toolbar_force)
        force_2d_layout.addWidget(self.canvas_force)

        # 滚动控制栏（时间窗口滑块）
        scroll_bar_layout = QHBoxLayout()
        scroll_bar_layout.setSpacing(6)
        lbl_window = QLabel("窗口宽度:")
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.5, 600.0)
        self.spin_window.setSingleStep(0.5)
        self.spin_window.setDecimals(1)
        self.spin_window.setValue(10.0)
        self.spin_window.setSuffix(" s")
        self.spin_window.setToolTip("可见时间窗口宽度")
        self.spin_window.valueChanged.connect(self._on_scroll_changed)
        self.scroll_force = QScrollBar(Qt.Horizontal)
        self.scroll_force.setRange(0, 0)
        self.scroll_force.valueChanged.connect(self._on_scroll_changed)
        btn_show_all = QPushButton("显示全部")
        btn_show_all.setObjectName("btnAddSeg")
        btn_show_all.clicked.connect(self._show_all_force)
        scroll_bar_layout.addWidget(lbl_window)
        scroll_bar_layout.addWidget(self.spin_window)
        scroll_bar_layout.addWidget(self.scroll_force, 1)
        scroll_bar_layout.addWidget(btn_show_all)
        force_2d_layout.addLayout(scroll_bar_layout)

        force_plot_layout.addWidget(force_2d_widget)
        main_layout.addWidget(force_plot_group, 2)

        # 力觉图表鼠标事件（拖拽有效段边界）
        self.canvas_force.mpl_connect('button_press_event', self._on_force_press)
        self.canvas_force.mpl_connect('motion_notify_event', self._on_force_motion)
        self.canvas_force.mpl_connect('button_release_event', self._on_force_release)

        # 滚动状态：None 表示显示全部
        self._force_view_all = True

        # ── 底部可折叠信息面板 (DockWidget) ──
        self.dock_info = QDockWidget("文件信息与数据检查", self)
        self.dock_info.setObjectName("dockInfo")
        self.dock_info.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.LeftDockWidgetArea)
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.dock_info.setWidget(self.txt_info)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_info)
        self.dock_info.hide()

        # 信息面板按钮 ↔ DockWidget 联动
        self.btn_toggle_info.clicked.connect(lambda checked: self.dock_info.setVisible(checked))
        self.dock_info.visibilityChanged.connect(self.btn_toggle_info.setChecked)

    # ──────────────────────────────────────────────
    #  时间偏移
    # ──────────────────────────────────────────────
    def _on_offset_changed(self, value):
        self.time_offset = value
        current_frame = self.spin_frame.value()
        self.plot_force_data(current_frame)

    # ──────────────────────────────────────────────
    #  文件打开 / 加载
    # ──────────────────────────────────────────────
    def _default_dir(self):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_calibration")
        return d if os.path.isdir(d) else ""

    def _update_file_list(self, filepath):
        """更新当前目录的文件列表"""
        self._current_dir = os.path.dirname(filepath)
        self._file_list = sorted([f for f in os.listdir(self._current_dir)
                                  if f.endswith(('.h5', '.hdf5'))])
        filename = os.path.basename(filepath)
        self._current_file_index = self._file_list.index(filename) if filename in self._file_list else -1
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """更新导航按钮状态"""
        self.btn_prev_file.setEnabled(self._current_file_index > 0)
        self.btn_next_file.setEnabled(self._current_file_index < len(self._file_list) - 1)

    def _load_prev_file(self):
        """加载上一个文件"""
        if self._current_file_index > 0:
            self._current_file_index -= 1
            filepath = os.path.join(self._current_dir, self._file_list[self._current_file_index])
            if self._is_from_processed:
                self._load_processed_file(filepath)
            else:
                self.load_file(filepath)
                self._auto_load_force_file(filepath)
            self._update_nav_buttons()

    def _load_next_file(self):
        """加载下一个文件"""
        if self._current_file_index < len(self._file_list) - 1:
            self._current_file_index += 1
            filepath = os.path.join(self._current_dir, self._file_list[self._current_file_index])
            if self._is_from_processed:
                self._load_processed_file(filepath)
            else:
                self.load_file(filepath)
                self._auto_load_force_file(filepath)
            self._update_nav_buttons()

    def _auto_load_force_file(self, filepath):
        """自动加载对应的力觉文件（根据时间戳就近匹配）"""
        basename = os.path.basename(filepath)
        dirpath = os.path.dirname(filepath)
        if not basename.startswith("calibration_") or basename.startswith("ft_"):
            return
        # 1) 尝试精确匹配: ft_ + 点云文件名
        ft_path = os.path.join(dirpath, "ft_" + basename)
        if os.path.isfile(ft_path):
            self.load_force_file(ft_path)
            self.statusBar().showMessage(
                self.statusBar().currentMessage() +
                f"  |  自动加载力觉文件: {os.path.basename(ft_path)}")
            return
        # 2) 扫描目录，按文件名中的时间戳就近匹配
        import glob
        import re
        ft_candidates = sorted(glob.glob(os.path.join(dirpath, "ft_calibration_*.h5")))
        if not ft_candidates:
            return
        # 从点云文件名提取时间戳
        m = re.search(r'(\d{8}_\d{6})', basename)
        if m:
            pc_ts = m.group(1)
            best, best_dist = None, float('inf')
            for cand in ft_candidates:
                cm = re.search(r'(\d{8}_\d{6})', os.path.basename(cand))
                if cm:
                    dist = abs(int(cm.group(1).replace('_', '')) - int(pc_ts.replace('_', '')))
                    if dist < best_dist:
                        best_dist = dist
                        best = cand
            if best is not None:
                self.load_force_file(best)
                self.statusBar().showMessage(
                    self.statusBar().currentMessage() +
                    f"  |  自动加载力觉文件: {os.path.basename(best)}")

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择点云 HDF5 文件", self._default_dir(),
            "HDF5 文件 (*.h5 *.hdf5);;所有文件 (*.*)")
        if not filepath:
            return
        self._update_file_list(filepath)
        self.load_file(filepath)
        self._auto_load_force_file(filepath)

    def open_force_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择力觉 HDF5 文件", self._default_dir(),
            "HDF5 文件 (*.h5 *.hdf5);;所有文件 (*.*)")
        if not filepath:
            return
        self.load_force_file(filepath)

    def load_force_file(self, filepath):
        """加载力觉数据文件"""
        self.ft_data = {}
        self._is_from_processed = False
        # 重置力偏差，避免旧偏差应用到新文件
        self.force_bias[:] = 0.0
        if hasattr(self, 'bias_spins'):
            for spin in self.bias_spins:
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)
        try:
            with h5py.File(filepath, 'r') as f:
                if 'force' not in f:
                    self.statusBar().showMessage(f"力觉文件中未找到 /force 组: {filepath}")
                    return
                grp = f['force']
                if 'timestamp' not in grp or 'values' not in grp:
                    self.statusBar().showMessage("力觉文件缺少 force/timestamp 或 force/values")
                    return
                self.ft_data['timestamp'] = grp['timestamp'][:]
                self.ft_data['values'] = grp['values'][:]
                cols = grp.attrs.get('columns', 'fx,fy,fz,mx,my,mz')
                if isinstance(cols, bytes):
                    cols = cols.decode('utf-8')
                self.ft_data['columns'] = cols.split(',')

        except Exception as e:
            self.statusBar().showMessage(f"读取力觉文件失败: {e}")
            return

        self._source_ft_path = filepath
        self.lbl_force_file.setText(filepath)

        # 信息面板追加摘要
        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values']
        lines = ["", "=" * 50, "  力觉数据文件信息", "=" * 50,
                 f"路径: {filepath}", f"采样数: {len(ft_ts)}",
                 f"通道: {self.ft_data['columns']}"]
        if len(ft_ts) > 1:
            dur = ft_ts[-1] - ft_ts[0]
            rate = (len(ft_ts) - 1) / dur if dur > 0 else 0
            lines += [f"总时长: {dur:.2f} s", f"采样率: {rate:.1f} Hz"]
        for i, col in enumerate(self.ft_data['columns']):
            v = ft_vals[:, i]
            lines.append(f"  {col}: min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")

        self.txt_info.append("\n".join(lines))
        self.statusBar().showMessage(f"已加载力觉文件: {os.path.basename(filepath)} | 采样数: {len(ft_ts)}")
        self._force_plot_cached = False
        self.plot_force_data()
        self.update_piezo_plot()

    def load_file(self, filepath):
        """加载点云标定数据文件"""
        self.txt_info.clear()
        self.h5_data = {}
        self.segments = []
        self._refresh_segment_list()
        self._is_from_processed = False
        lines = []

        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 ** 2:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / 1024 ** 2:.2f} MB"

        self._source_pc_path = filepath
        self.lbl_file.setText(filepath)
        lines += ["=" * 50, "  文件基本信息", "=" * 50,
                  f"路径: {filepath}", f"大小: {size_str}"]

        try:
            with h5py.File(filepath, 'r') as f:
                lines += ["", "=" * 50, "  文件结构", "=" * 50]

                def print_structure(name, obj):
                    indent = "  " * name.count("/")
                    if isinstance(obj, h5py.Group):
                        lines.append(f"{indent}/{name.split('/')[-1]}")
                        for k, v in obj.attrs.items():
                            lines.append(f"{indent}  @{k} = {v}")
                    elif isinstance(obj, h5py.Dataset):
                        lines.append(f"{indent}/{name.split('/')[-1]}  "
                                     f"shape={obj.shape}  dtype={obj.dtype}")
                f.visititems(print_structure)
                for k, v in f.attrs.items():
                    lines.append(f"  @{k} = {v}")

                # Meta 属性
                lines += ["", "=" * 50, "  /meta 属性", "=" * 50]
                if 'meta' in f:
                    meta = f['meta']
                    for k in meta.attrs:
                        lines.append(f"  {k} = {meta.attrs[k]}")
                    if len(meta.attrs) == 0:
                        lines.append("  (无属性)")
                else:
                    lines.append("  [!] /meta 组不存在")

                # 数据集维度检查
                lines += ["", "=" * 50, "  数据集维度检查", "=" * 50]
                expected = {
                    'vision/timestamp': {'ndim': 1, 'dtype_kind': 'f'},
                    'vision/xyz':       {'ndim': 3, 'dtype_kind': 'f'},
                    'vision/dxyz':      {'ndim': 3, 'dtype_kind': 'f'},
                    'vision/point_id':  {'ndim': 1, 'dtype_kind': 'i'},
                    'reference/xyz_ref': {'ndim': 2, 'dtype_kind': 'f'},
                }
                all_ok = True
                for path, spec in expected.items():
                    if path not in f:
                        lines.append(f"  [FAIL] {path} 不存在")
                        all_ok = False
                        continue
                    ds = f[path]
                    ok = (ds.ndim == spec['ndim'] and ds.dtype.kind == spec['dtype_kind'])
                    tag = " OK " if ok else "FAIL"
                    lines.append(f"  [{tag}] {path}: shape={ds.shape} dtype={ds.dtype}")
                    if not ok:
                        all_ok = False

                if 'vision/xyz' in f and 'vision/dxyz' in f:
                    match = f['vision/xyz'].shape == f['vision/dxyz'].shape
                    lines.append(f"  [{'  OK ' if match else 'FAIL'}] xyz 与 dxyz shape {'一致' if match else '不一致'}")
                    all_ok = all_ok and match

                if 'vision/xyz' in f and 'reference/xyz_ref' in f:
                    n1, n2 = f['vision/xyz'].shape[1], f['reference/xyz_ref'].shape[0]
                    match = n1 == n2
                    lines.append(f"  [{'  OK ' if match else 'FAIL'}] marker 数{'一致' if match else '不一致'}: N={n1}")
                    all_ok = all_ok and match

                if all_ok:
                    lines.append("  ── 全部通过 ──")

                # 数据有效性检查
                lines += ["", "=" * 50, "  数据有效性检查", "=" * 50]
                if 'vision/timestamp' in f:
                    ts = f['vision/timestamp'][:]
                    self.h5_data['timestamp'] = ts
                    if len(ts) > 1:
                        diffs = np.diff(ts)
                        mono = bool(np.all(diffs > 0))
                        lines.append(f"  [{'  OK ' if mono else 'FAIL'}] timestamp 单调递增: {mono}")
                        if mono:
                            lines.append(f"        帧间隔: min={diffs.min()*1000:.1f}ms  "
                                         f"max={diffs.max()*1000:.1f}ms  mean={diffs.mean()*1000:.1f}ms")
                            lines.append(f"        实际帧率: {1.0/diffs.mean():.1f} fps")
                            lines.append(f"        总时长: {ts[-1]-ts[0]:.2f} s")

                if 'vision/dxyz' in f:
                    dxyz = f['vision/dxyz'][:]
                    self.h5_data['dxyz'] = dxyz
                    first_max = np.max(np.abs(dxyz[0]))
                    is_zero = first_max < 1e-3
                    lines.append(f"  [{'  OK ' if is_zero else 'WARN'}] dxyz 首帧最大绝对值: {first_max:.6f}")
                    dxyz_norm = np.linalg.norm(dxyz, axis=2)
                    lines.append(f"        dxyz 位移范围: [{dxyz_norm.min():.4f}, {dxyz_norm.max():.4f}] mm")

                if 'vision/xyz' in f:
                    self.h5_data['xyz'] = f['vision/xyz'][:]
                if 'reference/xyz_ref' in f:
                    self.h5_data['xyz_ref'] = f['reference/xyz_ref'][:]
                if 'vision/point_id' in f:
                    self.h5_data['point_id'] = f['vision/point_id'][:]
                if 'vision/abnormal' in f:
                    self.h5_data['abnormal'] = f['vision/abnormal'][:]

        except Exception as e:
            lines.append(f"\n[ERROR] 读取文件失败: {e}")
            self.txt_info.setPlainText("\n".join(lines))
            return

        self.txt_info.setPlainText("\n".join(lines))

        # 加载压电数据（从点云文件自身）
        self._load_piezo_from_pc_file(filepath)

        # 压电数据信息追加到信息面板
        if self.piezo_data:
            piezo_lines = ["", "=" * 50, "  压电数据信息（来自点云文件）", "=" * 50,
                           f"ADC组: {self.piezo_data.get('adc_group', 'N/A')}",
                           f"采样数: {len(self.piezo_data['timestamp'])}",
                           f"通道数: {self.piezo_data.get('n_channels', 'N/A')}"]
            piezo_ts = self.piezo_data['timestamp']
            if len(piezo_ts) > 1:
                dur = piezo_ts[-1] - piezo_ts[0]
                rate = (len(piezo_ts) - 1) / dur if dur > 0 else 0
                piezo_lines += [f"时长: {dur:.2f} s", f"采样率: {rate:.1f} Hz"]
            piezo_vals = self.piezo_data['values']
            piezo_lines.append(f"电压范围 (CH{self.piezo_channel + 1}): {piezo_vals.min():.4f} ~ {piezo_vals.max():.4f} V")
            self.txt_info.append("\n".join(piezo_lines))

        if 'xyz' in self.h5_data:
            self.total_frames = self.h5_data['xyz'].shape[0]
            self.spin_frame.setRange(0, self.total_frames - 1)
            self.slider_frame.setRange(0, self.total_frames - 1)
            self.spin_frame.setValue(0)
            self.on_frame_changed(0)
            self.statusBar().showMessage(
                f"已加载: {os.path.basename(filepath)} | {size_str} | 共 {self.total_frames} 帧")
        else:
            self.statusBar().showMessage(f"已加载: {os.path.basename(filepath)} | {size_str}")

    # ──────────────────────────────────────────────
    #  帧切换 & 3D 可视化
    # ──────────────────────────────────────────────
    def on_frame_changed(self, frame_idx):
        if 'xyz' not in self.h5_data:
            return
        ts = self.h5_data.get('timestamp')
        if ts is not None:
            t_rel = ts[frame_idx] - ts[0]
            self.lbl_frame.setText(f"帧: {frame_idx} / {self.total_frames - 1}  |  t = {t_rel:.3f}s")
            self.statusBar().showMessage(
                f"帧 {frame_idx}/{self.total_frames - 1}  |  "
                f"相对时间: {t_rel:.3f}s  |  时间戳: {ts[frame_idx]:.6f}")
        else:
            self.lbl_frame.setText(f"帧: {frame_idx} / {self.total_frames - 1}")
        self.plot_frame(frame_idx)
        self.plot_force_data(frame_idx)
        self.update_piezo_plot(frame_idx)

    def plot_frame(self, frame_idx):
        xyz = self.h5_data['xyz']
        xyz_ref = self.h5_data.get('xyz_ref')
        pts = xyz[frame_idx]

        if xyz_ref is not None:
            disp = np.linalg.norm(pts - xyz_ref, axis=1)
        else:
            disp = np.zeros(len(pts))

        # 保存视角
        axes = self.fig.get_axes()
        if axes:
            ax_old = axes[0]
            if hasattr(ax_old, 'elev'):
                try:
                    self._saved_view = (ax_old.elev, ax_old.azim, getattr(ax_old, 'roll', None))
                except Exception:
                    pass

        self.fig.clear()
        # 手动指定 3D axes 位置，使其居中；colorbar 用独立 axes 避免挤压
        ax = self.fig.add_axes([0.05, 0.05, 0.78, 0.88], projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=disp, cmap='jet', s=50, vmin=0, vmax=2)
        cax = self.fig.add_axes([0.87, 0.15, 0.025, 0.6])
        cbar = self.fig.colorbar(sc, cax=cax)
        cbar.set_label('Displacement (mm)', rotation=270, labelpad=15)

        if xyz_ref is not None:
            ax.scatter(xyz_ref[:, 0], xyz_ref[:, 1], xyz_ref[:, 2],
                       c='gray', s=20, alpha=0.3, label='reference')
            ax.legend(loc='upper left', fontsize=8)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        ts = self.h5_data.get('timestamp')
        if ts is not None:
            ax.set_title(f'Frame {frame_idx}  |  t = {ts[frame_idx] - ts[0]:.3f}s  |  '
                         f'avg disp = {disp.mean():.3f}mm')
        else:
            ax.set_title(f'Frame {frame_idx}  |  avg disp = {disp.mean():.3f}mm')

        if len(pts) > 0:
            center = pts.mean(axis=0)
            half_range = max(np.ptp(pts[:, i]) for i in range(3)) / 2 * 1.1
            half_range = max(half_range, 0.1)
            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(center[1] - half_range, center[1] + half_range)
            ax.set_zlim(center[2] - half_range, center[2] + half_range)
            ax.set_box_aspect([1, 1, 1])

        if self._saved_view is not None:
            elev, azim, roll = self._saved_view
            try:
                ax.view_init(elev=elev, azim=azim, roll=roll)
            except TypeError:
                ax.view_init(elev=elev, azim=azim)
        else:
            try:
                ax.view_init(elev=90, azim=0, roll=-90)
            except TypeError:
                ax.view_init(elev=90, azim=0)

        self.canvas.draw()

    # ──────────────────────────────────────────────
    #  力觉数据滚动控制
    # ──────────────────────────────────────────────
    def _update_scroll_range(self):
        """根据力觉数据时长和窗口宽度更新滚动条范围"""
        if not self.ft_data:
            self.scroll_force.setRange(0, 0)
            return
        ft_ts = self.ft_data['timestamp']
        t_total = ft_ts[-1] - ft_ts[0]
        window = self.spin_window.value()
        if window >= t_total:
            self.scroll_force.setRange(0, 0)
        else:
            # 滚动条范围映射到毫秒，精度 0.1s
            max_val = int((t_total - window) * 10)
            self.scroll_force.setRange(0, max_val)
            self.scroll_force.setPageStep(int(window * 10))

    def _on_scroll_changed(self, _=None):
        self._force_view_all = False
        self._update_scroll_range()
        self.plot_force_data(self.spin_frame.value())

    def _show_all_force(self):
        self._force_view_all = True
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    # ──────────────────────────────────────────────
    #  力觉数据绘图
    # ──────────────────────────────────────────────
    def plot_force_data(self, frame_idx=None):
        if not self.ft_data:
            self.fig_force.clear()
            ax = self.fig_force.add_subplot(111)
            ax.text(0.5, 0.5, '未加载力觉数据\n请点击「打开力觉文件」',
                    ha='center', va='center', fontsize=12, color='#999999',
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_force.draw()
            self.fig_force3d.clear()
            self.canvas_force3d.draw()
            self._force_plot_cached = False
            return

        # 首次加载或数据变化时完整绘制，之后只更新帧标记
        if not self._force_plot_cached:
            self._plot_force_data_full(frame_idx)
            self._force_plot_cached = True
        else:
            self._update_force_frame_marker(frame_idx)

    def _plot_force_data_full(self, frame_idx=None):
        """完整绘制力觉图表（数据线、有效段等）"""
        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values'].copy()
        columns = self.ft_data['columns']
        n_ch = min(6, ft_vals.shape[1])
        ft_vals[:, :n_ch] -= self.force_bias[:n_ch]
        t_rel = ft_ts - ft_ts[0]
        t_total = t_rel[-1]

        self._update_scroll_range()
        self.fig_force.clear()
        self._force_frame_markers.clear()
        ax_f = self.fig_force.add_subplot(211)
        ax_m = self.fig_force.add_subplot(212, sharex=ax_f)

        colors_f = ['#E74C3C', '#2ECC71', '#3498DB']
        colors_m = ['#E67E22', '#9B59B6', '#1ABC9C']

        seg_ids = self.ft_data.get('segment_id')
        if seg_ids is not None:
            unique_segs = np.unique(seg_ids)
            for seg_i in unique_segs:
                mask = seg_ids == seg_i
                t_seg = t_rel[mask]
                vals_seg = ft_vals[mask]
                for i in range(min(3, ft_vals.shape[1])):
                    label = (columns[i] if i < len(columns) else f'ch{i}') if seg_i == unique_segs[0] else None
                    ax_f.plot(t_seg, vals_seg[:, i], color=colors_f[i], linewidth=0.8, label=label)
                for i in range(3, min(6, ft_vals.shape[1])):
                    label = (columns[i] if i < len(columns) else f'ch{i}') if seg_i == unique_segs[0] else None
                    ax_m.plot(t_seg, vals_seg[:, i], color=colors_m[i - 3], linewidth=0.8, label=label)
        else:
            for i in range(min(3, ft_vals.shape[1])):
                ax_f.plot(t_rel, ft_vals[:, i], color=colors_f[i], linewidth=0.8,
                          label=columns[i] if i < len(columns) else f'ch{i}')
            for i in range(3, min(6, ft_vals.shape[1])):
                ax_m.plot(t_rel, ft_vals[:, i], color=colors_m[i - 3], linewidth=0.8,
                          label=columns[i] if i < len(columns) else f'ch{i}')

        ax_f.set_ylabel('Force (N)')
        ax_f.legend(loc='upper right', fontsize=7, ncol=3)
        ax_f.grid(True, alpha=0.3)
        plt.setp(ax_f.get_xticklabels(), visible=False)

        ax_m.set_ylabel('Torque (Nm)')
        ax_m.set_xlabel('Time (s)')
        ax_m.legend(loc='upper right', fontsize=7, ncol=3)
        ax_m.grid(True, alpha=0.3)

        self._seg_artists = []
        self._force_axes = (ax_f, ax_m)
        sel_row = self.list_segments.currentRow()
        for i, (s, e) in enumerate(self.segments):
            is_sel = (i == sel_row)
            color = '#F39C12' if is_sel else '#2ECC71'
            alpha = 0.25 if is_sel else 0.15
            span_f = ax_f.axvspan(s, e, color=color, alpha=alpha)
            span_m = ax_m.axvspan(s, e, color=color, alpha=alpha)
            vl_s_f = ax_f.axvline(s, color=color, linewidth=1.5, linestyle='-', alpha=0.6)
            vl_s_m = ax_m.axvline(s, color=color, linewidth=1.5, linestyle='-', alpha=0.6)
            vl_e_f = ax_f.axvline(e, color=color, linewidth=1.5, linestyle='-', alpha=0.6)
            vl_e_m = ax_m.axvline(e, color=color, linewidth=1.5, linestyle='-', alpha=0.6)
            self._seg_artists.append((span_f, span_m, vl_s_f, vl_s_m, vl_e_f, vl_e_m))

        # 绘制异常帧标记
        if 'abnormal' in self.h5_data and 'timestamp' in self.h5_data:
            abnormal = self.h5_data['abnormal']
            vision_ts = self.h5_data['timestamp']
            abnormal_indices = np.where(abnormal)[0]
            for ab_idx in abnormal_indices:
                ab_ft_rel = vision_ts[ab_idx] + self.time_offset - ft_ts[0]
                for ax in [ax_f, ax_m]:
                    ax.axvline(ab_ft_rel, color='#E74C3C', linewidth=1.0,
                               linestyle=':', alpha=0.7)
            if len(abnormal_indices) > 0:
                ax_f.axvline(np.nan, color='#E74C3C', linewidth=1.0,
                             linestyle=':', alpha=0.7, label=f'异常帧 ({len(abnormal_indices)})')
                ax_f.legend(loc='upper right', fontsize=7, ncol=4)

        # 防止数据全为零时 Y 轴范围过小
        for ax, min_half in [(ax_f, 5.0), (ax_m, 0.5)]:
            ylo, yhi = ax.get_ylim()
            if yhi - ylo < min_half * 2:
                mid = (ylo + yhi) / 2
                ax.set_ylim(mid - min_half, mid + min_half)

        self.fig_force.subplots_adjust(
            left=0.06, right=0.92, top=0.92, bottom=0.12, hspace=0.25)
        self.canvas_force.draw()

        self._update_force_frame_marker(frame_idx)

    def _update_force_frame_marker(self, frame_idx=None):
        """只更新当前帧标记和3D显示，不重新绘制数据线"""
        if not self.ft_data or self._force_axes is None:
            return

        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values'].copy()
        columns = self.ft_data['columns']
        n_ch = min(6, ft_vals.shape[1])
        ft_vals[:, :n_ch] -= self.force_bias[:n_ch]
        t_rel = ft_ts - ft_ts[0]
        t_total = t_rel[-1]

        ax_f, ax_m = self._force_axes
        cur_force_vals = None

        # 清除旧的帧标记线（直接移除已追踪的标记线）
        for marker in self._force_frame_markers:
            marker.remove()
        self._force_frame_markers.clear()

        # 绘制当前帧竖线
        if frame_idx is not None and 'timestamp' in self.h5_data:
            vision_ts = self.h5_data['timestamp']
            cur_abs = vision_ts[frame_idx] + self.time_offset
            cur_ft_rel = cur_abs - ft_ts[0]
            m1 = ax_f.axvline(cur_ft_rel, color='#333333', linewidth=1.2,
                        linestyle='--', alpha=0.8)
            m2 = ax_m.axvline(cur_ft_rel, color='#333333', linewidth=1.2,
                        linestyle='--', alpha=0.8)
            self._force_frame_markers.extend([m1, m2])
            if ft_ts[0] <= cur_abs <= ft_ts[-1]:
                idx = min(np.searchsorted(ft_ts, cur_abs), len(ft_ts) - 1)
                cur_force_vals = ft_vals[idx]
                val_str = '  '.join(f'{columns[j]}={cur_force_vals[j]:.2f}' for j in range(min(6, len(columns))))
                ax_f.set_title(f't={cur_ft_rel:.3f}s  |  {val_str}', fontsize=9)

        # 应用时间窗口
        if not self._force_view_all:
            window = self.spin_window.value()
            if window < t_total:
                t_start = self.scroll_force.value() / 10.0
                t_end = t_start + window
                ax_f.set_xlim(t_start, t_end)

        self.canvas_force.draw()
        self._plot_force_3d(cur_force_vals, columns)

    # ──────────────────────────────────────────────
    #  3D 力向量可视化
    # ──────────────────────────────────────────────
    def _plot_force_3d(self, cur_force_vals, columns):
        """在右侧 3D 图中显示当前帧的力和力矩向量"""
        # 保存视角
        axes = self.fig_force3d.get_axes()
        if axes:
            ax_old = axes[0]
            if hasattr(ax_old, 'elev'):
                try:
                    self._saved_force3d_view = (ax_old.elev, ax_old.azim, getattr(ax_old, 'roll', None))
                except Exception:
                    pass

        self.fig_force3d.clear()

        if cur_force_vals is None or len(cur_force_vals) < 3:
            ax = self.fig_force3d.add_subplot(111)
            ax.text(0.5, 0.5, '无当前帧\n力觉数据',
                    ha='center', va='center', fontsize=10, color='#999999',
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_force3d.draw()
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax3 = self.fig_force3d.add_subplot(111, projection='3d')

        origin = [0, 0, 0]
        fx, fy, fz = cur_force_vals[0], cur_force_vals[1], cur_force_vals[2]

        # 绘制力向量 (红色箭头)
        ax3.quiver(*origin, fx, fy, fz, color='#E74C3C', arrow_length_ratio=0.15,
                   linewidth=2.0, label='Force (N)')

        # 绘制力矩向量 (蓝色箭头)
        if len(cur_force_vals) >= 6:
            mx, my, mz = cur_force_vals[3], cur_force_vals[4], cur_force_vals[5]
            ax3.quiver(*origin, mx, my, mz, color='#3498DB', arrow_length_ratio=0.15,
                       linewidth=2.0, label='Torque (Nm)')
            all_vals = [abs(v) for v in [fx, fy, fz, mx, my, mz]]
        else:
            all_vals = [abs(v) for v in [fx, fy, fz]]

        # 设置坐标范围（对称，以原点为中心）
        max_range = max(max(all_vals) * 1.3, 0.1)
        ax3.set_xlim(-max_range, max_range)
        ax3.set_ylim(-max_range, max_range)
        ax3.set_zlim(-max_range, max_range)

        ax3.set_xlabel('X', fontsize=8)
        ax3.set_ylabel('Y', fontsize=8)
        ax3.set_zlabel('Z', fontsize=8)
        ax3.tick_params(labelsize=6)

        # 标注数值
        force_str = f'F=({fx:.1f}, {fy:.1f}, {fz:.1f})'
        if len(cur_force_vals) >= 6:
            force_str += f'\nM=({mx:.1f}, {my:.1f}, {mz:.1f})'
        ax3.set_title(force_str, fontsize=8, pad=2)
        ax3.legend(loc='upper left', fontsize=6)

        try:
            ax3.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        # 恢复视角
        if self._saved_force3d_view is not None:
            elev, azim, roll = self._saved_force3d_view
            try:
                ax3.view_init(elev=elev, azim=azim, roll=roll)
            except TypeError:
                ax3.view_init(elev=elev, azim=azim)

        self.fig_force3d.tight_layout()
        self.canvas_force3d.draw()

    # ──────────────────────────────────────────────
    #  有效段管理
    # ──────────────────────────────────────────────
    def _auto_detect_segments(self):
        """自动检测有效片段：寻峰法 — 找 fz 所有显著负峰，围绕每个峰扩展段边界"""
        if not self.ft_data:
            QMessageBox.warning(self, "提示", "请先加载力觉数据")
            return

        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values'].copy()
        n_ch = min(6, ft_vals.shape[1])
        ft_vals[:, :n_ch] -= self.force_bias[:n_ch]
        t_rel = ft_ts - ft_ts[0]

        # fz 是第3个通道 (index 2)
        fz = ft_vals[:, 2]
        n = len(fz)

        # ── 采样率 ──
        if len(ft_ts) > 1:
            sample_rate = (len(ft_ts) - 1) / (ft_ts[-1] - ft_ts[0])
        else:
            sample_rate = 100.0

        # ── 平滑 fz（窗口约 0.05s，仅去高频噪声） ──
        kernel_size = max(3, int(sample_rate * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        fz_smooth = np.convolve(fz, kernel, mode='same')

        # ── 自适应阈值 ──
        fz_min = fz_smooth.min()
        # 峰值阈值：最小值的 10%，保证浅峰也能检出，但不低于 -0.5N
        threshold_peak = max(fz_min * 0.10, -0.5)
        # 边界阈值：峰值阈值的 15%（接近零）
        threshold_edge = threshold_peak * 0.15

        # ── 寻找所有局部极小值（负峰） ──
        # 不逐点比较，改用"区间最小值"：找 fz 低于阈值的连续区间，取区间内最小点
        peak_indices = []
        i = 0
        while i < n:
            if fz_smooth[i] < threshold_peak:
                # 进入一个低于阈值的连续区间
                region_start = i
                min_val = fz_smooth[i]
                min_idx = i
                while i < n and fz_smooth[i] < threshold_peak:
                    if fz_smooth[i] < min_val:
                        min_val = fz_smooth[i]
                        min_idx = i
                    i += 1
                peak_indices.append(min_idx)
            else:
                i += 1

        # ── 合并过于接近的峰（间距 < 0.3s，取更深的） ──
        if len(peak_indices) > 1:
            min_peak_dist = int(sample_rate * 0.3)
            merged_peaks = [peak_indices[0]]
            for pk in peak_indices[1:]:
                if pk - merged_peaks[-1] < min_peak_dist:
                    if fz_smooth[pk] < fz_smooth[merged_peaks[-1]]:
                        merged_peaks[-1] = pk
                else:
                    merged_peaks.append(pk)
            peak_indices = merged_peaks

        # ── 围绕每个峰扩展段边界 ──
        detected = []
        for idx, pk in enumerate(peak_indices):
            # 确定搜索范围：不越过相邻峰之间的局部最大值
            if idx == 0:
                search_left_limit = 0
            else:
                # 找前一个峰和当前峰之间的局部最大值（休息点）
                prev_pk = peak_indices[idx - 1]
                region = fz_smooth[prev_pk:pk + 1]
                local_max_offset = int(np.argmax(region))
                search_left_limit = prev_pk + local_max_offset

            # 向左找边界：从峰向左，直到 fz 回升到 threshold_edge 以上，
            # 但不超过 search_left_limit
            left = pk
            while left > search_left_limit and fz_smooth[left - 1] < threshold_edge:
                left -= 1

            # 只取前半段（从开始下降到极小值）
            detected.append((t_rel[left], t_rel[pk]))

        # ── 去除重叠段 ──
        if len(detected) > 1:
            non_overlap = [detected[0]]
            for s, e in detected[1:]:
                if s > non_overlap[-1][1]:
                    non_overlap.append((s, e))
                else:
                    # 重叠时保留更长的段
                    prev_s, prev_e = non_overlap[-1]
                    if (e - s) > (prev_e - prev_s):
                        non_overlap[-1] = (s, e)
            detected = non_overlap

        if not detected:
            QMessageBox.information(self, "提示", "未检测到有效片段 (fz 极小值 < -1N)")
            return

        # 过滤包含异常帧的片段
        abnormal = self.h5_data.get('abnormal')
        vision_ts = self.h5_data.get('timestamp')
        filtered = []
        skipped = 0
        if abnormal is not None and vision_ts is not None:
            ab_indices = np.where(abnormal)[0]
            if len(ab_indices) > 0:
                # 异常帧的力觉相对时间
                ab_times = vision_ts[ab_indices] + self.time_offset - ft_ts[0]
                for s, e in detected:
                    has_abnormal = np.any((ab_times >= s) & (ab_times <= e))
                    if has_abnormal:
                        skipped += 1
                    else:
                        filtered.append([s, e])
            else:
                filtered = [[s, e] for s, e in detected]
        else:
            filtered = [[s, e] for s, e in detected]

        self.segments = filtered
        self._refresh_segment_list()
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

        msg = f"检测到 {len(detected)} 个片段"
        if skipped > 0:
            msg += f"，其中 {skipped} 个包含异常帧已排除"
        msg += f"\n最终保留 {len(filtered)} 个有效片段，可拖拽边界微调"
        QMessageBox.information(self, "自动检测完成", msg)

    def _auto_detect_press_segments(self):
        """自动检测完整按压过程：从按下开始直至释放完成"""
        if not self.ft_data:
            QMessageBox.warning(self, "提示", "请先加载力觉数据")
            return

        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values'].copy()
        n_ch = min(6, ft_vals.shape[1])
        ft_vals[:, :n_ch] -= self.force_bias[:n_ch]
        t_rel = ft_ts - ft_ts[0]

        fz = ft_vals[:, 2]
        n = len(fz)

        if len(ft_ts) > 1:
            sample_rate = (len(ft_ts) - 1) / (ft_ts[-1] - ft_ts[0])
        else:
            sample_rate = 100.0

        kernel_size = max(3, int(sample_rate * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        fz_smooth = np.convolve(fz, kernel, mode='same')

        fz_min = fz_smooth.min()
        threshold_peak = max(fz_min * 0.10, -0.5)
        threshold_edge = threshold_peak * 0.15

        # 寻找所有负峰（与自动选择有效段相同）
        peak_indices = []
        i = 0
        while i < n:
            if fz_smooth[i] < threshold_peak:
                min_val = fz_smooth[i]
                min_idx = i
                while i < n and fz_smooth[i] < threshold_peak:
                    if fz_smooth[i] < min_val:
                        min_val = fz_smooth[i]
                        min_idx = i
                    i += 1
                peak_indices.append(min_idx)
            else:
                i += 1

        if len(peak_indices) > 1:
            min_peak_dist = int(sample_rate * 0.3)
            merged_peaks = [peak_indices[0]]
            for pk in peak_indices[1:]:
                if pk - merged_peaks[-1] < min_peak_dist:
                    if fz_smooth[pk] < fz_smooth[merged_peaks[-1]]:
                        merged_peaks[-1] = pk
                else:
                    merged_peaks.append(pk)
            peak_indices = merged_peaks

        detected = []
        for idx, pk in enumerate(peak_indices):
            # 左侧搜索范围：不越过前一个峰与当前峰之间的局部最大值
            if idx == 0:
                search_left_limit = 0
            else:
                prev_pk = peak_indices[idx - 1]
                region = fz_smooth[prev_pk:pk + 1]
                search_left_limit = prev_pk + int(np.argmax(region))

            # 右侧搜索范围：不越过当前峰与下一个峰之间的局部最大值
            if idx == len(peak_indices) - 1:
                search_right_limit = n - 1
            else:
                next_pk = peak_indices[idx + 1]
                region = fz_smooth[pk:next_pk + 1]
                search_right_limit = pk + int(np.argmax(region))

            # 向左找按下起点
            left = pk
            while left > search_left_limit and fz_smooth[left - 1] < threshold_edge:
                left -= 1

            # 向右找释放终点
            right = pk
            while right < search_right_limit and fz_smooth[right + 1] < threshold_edge:
                right += 1

            detected.append((t_rel[left], t_rel[right]))

        # 去除重叠段
        if len(detected) > 1:
            non_overlap = [detected[0]]
            for s, e in detected[1:]:
                if s > non_overlap[-1][1]:
                    non_overlap.append((s, e))
                else:
                    prev_s, prev_e = non_overlap[-1]
                    if (e - s) > (prev_e - prev_s):
                        non_overlap[-1] = (s, e)
            detected = non_overlap

        if not detected:
            QMessageBox.information(self, "提示", "未检测到按压过程 (fz 极小值 < -1N)")
            return

        # 过滤包含异常帧的片段
        abnormal = self.h5_data.get('abnormal')
        vision_ts = self.h5_data.get('timestamp')
        filtered = []
        skipped = 0
        if abnormal is not None and vision_ts is not None:
            ab_indices = np.where(abnormal)[0]
            if len(ab_indices) > 0:
                ab_times = vision_ts[ab_indices] + self.time_offset - ft_ts[0]
                for s, e in detected:
                    if np.any((ab_times >= s) & (ab_times <= e)):
                        skipped += 1
                    else:
                        filtered.append([s, e])
            else:
                filtered = [[s, e] for s, e in detected]
        else:
            filtered = [[s, e] for s, e in detected]

        self.segments = filtered
        self._refresh_segment_list()
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

        msg = f"检测到 {len(detected)} 个按压过程"
        if skipped > 0:
            msg += f"，其中 {skipped} 个包含异常帧已排除"
        msg += f"\n最终保留 {len(filtered)} 个完整按压段，可拖拽边界微调"
        QMessageBox.information(self, "自动检测完成", msg)

    def _add_segment(self):
        if not self.ft_data:
            QMessageBox.warning(self, "提示", "请先加载力觉数据")
            return
        ft_ts = self.ft_data['timestamp']
        t_range = ft_ts[-1] - ft_ts[0]
        seg_len = min(4.0, t_range * 0.8)  # 默认4秒，不超过总时长80%
        gap = 0.5  # 段间间隔
        if len(self.segments) == 0:
            start = t_range * 0.05
            end = min(start + seg_len, t_range)
        else:
            last_end = self.segments[-1][1]
            start = last_end + gap
            end = min(start + seg_len, t_range)
            if start >= t_range:
                start = max(t_range - seg_len, 0)
                end = t_range
        self.segments.append([start, end])
        self._refresh_segment_list()
        self.list_segments.setCurrentRow(len(self.segments) - 1)
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    def _del_segment(self):
        row = self.list_segments.currentRow()
        if 0 <= row < len(self.segments):
            self.segments.pop(row)
            self._refresh_segment_list()
            self._force_plot_cached = False
            self.plot_force_data(self.spin_frame.value())

    def _refresh_segment_list(self):
        self.list_segments.clear()
        for i, (s, e) in enumerate(self.segments):
            self.list_segments.addItem(f"段 {i+1}: {s:.3f}s ~ {e:.3f}s")

    def _on_segment_selected(self, row):
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    # ── 鼠标拖拽有效段边界 ──
    def _on_force_press(self, event):
        if event.inaxes is None or not self.segments or not self.ft_data:
            return
        x = event.xdata
        xlim = event.inaxes.get_xlim()
        visible_range = xlim[1] - xlim[0]
        threshold = visible_range * 0.02

        # 优先检查是否点击边界（用于拖拽）
        best_dist = threshold
        best_hit = None
        for i, (s, e) in enumerate(self.segments):
            d_left = abs(x - s)
            if d_left < best_dist:
                best_dist = d_left
                best_hit = (i, 'left')
            d_right = abs(x - e)
            if d_right < best_dist:
                best_dist = d_right
                best_hit = (i, 'right')

        if best_hit is not None:
            self._dragging_segment = best_hit
        else:
            # 未点击边界，检查是否点击段内部（用于选中）
            for i, (s, e) in enumerate(self.segments):
                if s <= x <= e:
                    self.list_segments.setCurrentRow(i)
                    self._force_plot_cached = False
                    self.plot_force_data(self.spin_frame.value())
                    break

    def _on_force_motion(self, event):
        if self._dragging_segment is None or event.inaxes is None:
            return
        idx, side = self._dragging_segment
        x = event.xdata
        seg = self.segments[idx]
        if side == 'left':
            seg[0] = min(x, seg[1] - 0.01)
        else:
            seg[1] = max(x, seg[0] + 0.01)
        self._refresh_segment_list()
        self.list_segments.blockSignals(True)
        self.list_segments.setCurrentRow(idx)
        self.list_segments.blockSignals(False)
        # 快速更新：仅移动当前段的 artist，不重绘整个图
        if idx < len(self._seg_artists) and self._force_axes is not None:
            s, e = seg
            span_f, span_m, vl_s_f, vl_s_m, vl_e_f, vl_e_m = self._seg_artists[idx]
            # 更新 axvspan 范围
            for span in (span_f, span_m):
                xy = span.get_xy()
                # axvspan 的 xy 是 [(x0,y0),(x0,y1),(x1,y1),(x1,y0),(x0,y0)]
                xy[0][0] = s; xy[1][0] = s; xy[4][0] = s
                xy[2][0] = e; xy[3][0] = e
                span.set_xy(xy)
            # 更新 axvline 位置
            for vl in (vl_s_f, vl_s_m):
                vl.set_xdata([s, s])
            for vl in (vl_e_f, vl_e_m):
                vl.set_xdata([e, e])
            self.canvas_force.draw_idle()
        else:
            self.plot_force_data(self.spin_frame.value())

    def _on_force_release(self, event):
        if self._dragging_segment is not None:
            self._dragging_segment = None
            self._force_plot_cached = False
            self.plot_force_data(self.spin_frame.value())

    # ──────────────────────────────────────────────
    #  力传感器静态偏差
    # ──────────────────────────────────────────────
    def _on_bias_changed(self, _=None):
        for i, spin in enumerate(self.bias_spins):
            self.force_bias[i] = spin.value()
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    def _auto_zero_bias(self):
        if not self.ft_data:
            QMessageBox.warning(self, "提示", "请先加载力觉数据")
            return
        vals = self.ft_data['values']
        n = min(50, len(vals))
        mean_vals = vals[:n].mean(axis=0)
        for spin in self.bias_spins:
            spin.blockSignals(True)
        for i in range(min(6, len(mean_vals))):
            self.bias_spins[i].setValue(float(mean_vals[i]))
            self.force_bias[i] = float(mean_vals[i])
        for spin in self.bias_spins:
            spin.blockSignals(False)
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    def _reset_bias(self):
        for spin in self.bias_spins:
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self.force_bias[:] = 0.0
        self._force_plot_cached = False
        self.plot_force_data(self.spin_frame.value())

    # ──────────────────────────────────────────────
    #  压电信号相关方法
    # ──────────────────────────────────────────────
    def _load_piezo_from_pc_file(self, filepath):
        """从点云文件中加载压电数据（/piezo_stream/adc{group}/）"""
        self.piezo_data = {}
        try:
            with h5py.File(filepath, 'r') as f:
                if 'piezo_stream' not in f:
                    return
                ps = f['piezo_stream']
                grp_name = f'adc{self.piezo_adc_group + 1}'
                if grp_name not in ps:
                    return
                grp = ps[grp_name]
                if 'timestamp' not in grp or 'values' not in grp:
                    return
                ts = grp['timestamp'][:]
                raw_vals = grp['values'][:]  # (N, 8)
                self.piezo_data['timestamp'] = ts
                self.piezo_data['raw_values'] = raw_vals
                if raw_vals.ndim == 2 and raw_vals.shape[1] == 8:
                    self.piezo_data['values'] = raw_vals[:, self.piezo_channel]
                    self.piezo_data['n_channels'] = 8
                else:
                    self.piezo_data['values'] = raw_vals
                    self.piezo_data['n_channels'] = 1
                self.piezo_data['channel'] = self.piezo_channel + 1
                self.piezo_data['adc_group'] = self.piezo_adc_group + 1
                print(f"已从点云文件加载压电数据 (ADC{self.piezo_adc_group + 1}): "
                      f"{len(ts)} 采样, CH{self.piezo_channel + 1}")
        except Exception as e:
            print(f"加载压电数据失败: {e}")

    def _on_piezo_adc_group_changed(self, index):
        """ADC组切换：重新从点云文件加载对应ADC组的压电数据"""
        self.piezo_adc_group = index
        if self._source_pc_path:
            self._load_piezo_from_pc_file(self._source_pc_path)
        self.update_piezo_plot()

    def _on_piezo_channel_changed(self, index):
        """压电通道切换：从已加载的8通道原始数据中取对应列"""
        self.piezo_channel = index
        if self.piezo_data and 'raw_values' in self.piezo_data:
            raw = self.piezo_data['raw_values']
            if raw.ndim == 2 and index < raw.shape[1]:
                self.piezo_data['values'] = raw[:, index]
            elif raw.ndim == 1:
                self.piezo_data['values'] = raw
        self.update_piezo_plot()

    def _on_piezo_window_changed(self, value):
        """时间窗口大小改变"""
        self.piezo_window_ms = value

    def update_piezo_plot(self, frame_idx=None):
        """更新压电波形显示 (matplotlib)"""
        if not self.piezo_data:
            self.piezo_line.set_data([], [])
            self.piezo_marker.set_xdata([0, 0])
            self.piezo_ax.set_title("压电信号 (无数据)", fontsize=10)
            self.piezo_ax.relim()
            self.piezo_ax.autoscale_view()
            self.canvas_piezo.draw_idle()
            return

        piezo_ts = self.piezo_data['timestamp']
        piezo_vals_all = np.asarray(self.piezo_data['values'], dtype=np.float64).ravel()
        adc_group = self.piezo_data.get('adc_group', self.piezo_adc_group + 1)
        ch = self.piezo_channel + 1

        # 过滤 NaN/Inf，计算全量数据的Y范围
        finite_mask = np.isfinite(piezo_vals_all)
        if not np.any(finite_mask):
            self.piezo_line.set_data([], [])
            self.piezo_marker.set_xdata([0, 0])
            self.piezo_ax.set_title(f"压电信号 — ADC{adc_group} CH{ch} (无效数据)", fontsize=10)
            self.canvas_piezo.draw_idle()
            return

        y_min = float(piezo_vals_all[finite_mask].min())
        y_max = float(piezo_vals_all[finite_mask].max())
        y_span = y_max - y_min
        if y_span < 1e-6:
            self.piezo_ax.set_ylim(y_min - 0.5, y_max + 0.5)
        else:
            margin = max(y_span * 0.1, 0.05)
            self.piezo_ax.set_ylim(y_min - margin, y_max + margin)

        # 计算相对时间
        t0 = piezo_ts[0]
        t_rel_all = (piezo_ts - t0).astype(np.float64)

        # 大数据降采样显示（保留波形形状）
        n = len(t_rel_all)
        MAX_DISPLAY = 50000
        if n > MAX_DISPLAY:
            stride = max(1, n // MAX_DISPLAY)
            t_plot = t_rel_all[::stride]
            v_plot = piezo_vals_all[::stride]
        else:
            t_plot = t_rel_all
            v_plot = piezo_vals_all

        self.piezo_line.set_data(t_plot, v_plot)
        self.piezo_ax.set_xlim(t_rel_all[0], t_rel_all[-1])
        self.piezo_ax.set_title(f"压电信号 — ADC{adc_group} CH{ch}", fontsize=10)

        # 更新帧标记线
        if frame_idx is not None and 'timestamp' in self.h5_data:
            vision_ts = self.h5_data['timestamp']
            cur_abs = vision_ts[frame_idx] + self.time_offset
            cur_piezo_rel = cur_abs - t0
        else:
            cur_piezo_rel = t_rel_all[0]
        self.piezo_marker.set_xdata([cur_piezo_rel, cur_piezo_rel])

        self.canvas_piezo.draw_idle()

    def extract_piezo_window_features(self, vision_timestamp, piezo_ts, piezo_vals, window_ms=33):
        """提取时间窗口 [T-window_ms, T] 内的压电统计特征

        Args:
            vision_timestamp: 视觉帧绝对时间戳
            piezo_ts: 压电时间戳数组
            piezo_vals: 压电电压值数组
            window_ms: 时间窗口大小（毫秒）

        Returns:
            np.ndarray, shape (5,): [mean, std, rms, max, energy]
        """
        window_sec = window_ms / 1000.0
        t_start = vision_timestamp - window_sec
        t_end = vision_timestamp

        mask = (piezo_ts >= t_start) & (piezo_ts <= t_end)
        window_vals = piezo_vals[mask]

        if len(window_vals) == 0:
            return np.zeros(5, dtype=np.float32)

        window_vals = window_vals.astype(np.float32)
        return np.array([
            np.mean(window_vals),
            np.std(window_vals),
            np.sqrt(np.mean(window_vals ** 2)),
            np.max(np.abs(window_vals)),
            np.sum(np.abs(window_vals))
        ], dtype=np.float32)

    # ──────────────────────────────────────────────
    #  保存处理数据
    # ──────────────────────────────────────────────
    def _save_processed(self):
        if 'xyz' not in self.h5_data:
            QMessageBox.warning(self, "警告", "请先加载点云数据")
            return
        if not self.ft_data:
            QMessageBox.warning(self, "警告", "请先加载力觉数据")
            return
        if not self.segments:
            QMessageBox.warning(self, "警告", "请先添加有效数据段")
            return

        # 自动命名：processed_<原文件名>，保存到原文件所在目录
        save_dir = os.path.dirname(self._source_pc_path) if self._source_pc_path else self._default_dir()
        base = os.path.basename(self._source_pc_path) if self._source_pc_path else "data.h5"
        name, ext = os.path.splitext(base)
        # 避免反复叠加 processed_ 前缀
        if not name.startswith("processed_"):
            name = f"processed_{name}"
        filepath = os.path.join(save_dir, f"{name}{ext}")
        # 若同名文件已存在，追加序号
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(save_dir, f"{name}_{counter}{ext}")
            counter += 1

        vision_ts = self.h5_data['timestamp']
        ft_ts = self.ft_data['timestamp']
        ft_vals = self.ft_data['values']

        if self._is_from_processed:
            # ── 再编辑模式：force 与 vision 已经 1:1 对齐 ──
            # 段的时间坐标是相对于 ft_ts[0] 的
            ft_t0 = ft_ts[0]
            # vision timestamp 和 force timestamp 相同（已对齐），用 vision_ts 做段筛选
            t_rel = vision_ts - ft_t0

            valid_mask = np.zeros(len(vision_ts), dtype=bool)
            seg_id_arr = np.full(len(vision_ts), -1, dtype=np.int32)
            for seg_i, (s_rel, e_rel) in enumerate(self.segments):
                in_seg = (t_rel >= s_rel) & (t_rel <= e_rel)
                valid_mask |= in_seg
                seg_id_arr[in_seg] = seg_i

            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) == 0:
                QMessageBox.warning(self, "警告", "有效段内无点云帧，请调整段范围")
                return

            # 直接取已有力数据（不重新插值），并应用偏差
            save_force = ft_vals[valid_idx].copy()
            n_ch = min(6, save_force.shape[1])
            save_force[:, :n_ch] -= self.force_bias[:n_ch]
            save_ts = vision_ts[valid_idx]
        else:
            # ── 原始模式：需要对齐 + 插值 ──
            aligned_vision_ts = vision_ts + self.time_offset
            ft_t0 = ft_ts[0]

            valid_mask = np.zeros(len(vision_ts), dtype=bool)
            seg_id_arr = np.full(len(vision_ts), -1, dtype=np.int32)
            for seg_i, (s_rel, e_rel) in enumerate(self.segments):
                s_abs = ft_t0 + s_rel
                e_abs = ft_t0 + e_rel
                in_seg = (aligned_vision_ts >= s_abs) & (aligned_vision_ts <= e_abs)
                valid_mask |= in_seg
                seg_id_arr[in_seg] = seg_i

            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) == 0:
                QMessageBox.warning(self, "警告", "有效段内无点云帧，请调整段范围或时间偏移")
                return

            # 插值力觉数据到每个有效帧的对齐时间戳，并应用偏差
            interp_force = np.zeros((len(valid_idx), ft_vals.shape[1]), dtype=np.float64)
            for ch in range(ft_vals.shape[1]):
                interp_force[:, ch] = np.interp(
                    aligned_vision_ts[valid_idx], ft_ts, ft_vals[:, ch])
            n_ch = min(6, interp_force.shape[1])
            interp_force[:, :n_ch] -= self.force_bias[:n_ch]
            save_force = interp_force
            save_ts = aligned_vision_ts[valid_idx]

        try:
            with h5py.File(filepath, 'w') as f:
                vg = f.create_group('vision')
                vg.create_dataset('timestamp', data=vision_ts[valid_idx])
                vg.create_dataset('xyz', data=self.h5_data['xyz'][valid_idx])
                if 'dxyz' in self.h5_data:
                    vg.create_dataset('dxyz', data=self.h5_data['dxyz'][valid_idx])
                if 'point_id' in self.h5_data:
                    vg.create_dataset('point_id', data=self.h5_data['point_id'])

                rg = f.create_group('reference')
                if 'xyz_ref' in self.h5_data:
                    rg.create_dataset('xyz_ref', data=self.h5_data['xyz_ref'])

                fg = f.create_group('force')
                fg.create_dataset('timestamp', data=save_ts)
                fg.create_dataset('values', data=save_force)
                fg.create_dataset('segment_id', data=seg_id_arr[valid_idx])
                fg.attrs['columns'] = ','.join(self.ft_data['columns'])

                # 提取并保存压电特征（如果有压电数据）
                if self.piezo_data:
                    piezo_ts = self.piezo_data['timestamp']
                    # 使用当前选中通道的数据
                    if 'raw_values' in self.piezo_data:
                        raw = self.piezo_data['raw_values']
                        if raw.ndim == 2 and self.piezo_channel < raw.shape[1]:
                            piezo_vals = raw[:, self.piezo_channel]
                        else:
                            piezo_vals = self.piezo_data['values']
                    else:
                        piezo_vals = self.piezo_data['values']
                    window_ms = self.piezo_window_ms

                    piezo_features = []
                    for ts in save_ts:
                        feat = self.extract_piezo_window_features(ts, piezo_ts, piezo_vals, window_ms)
                        piezo_features.append(feat)
                    piezo_features = np.array(piezo_features, dtype=np.float32)

                    pg_grp = f.create_group('piezo')
                    pg_grp.create_dataset('features', data=piezo_features)
                    pg_grp.attrs['feature_names'] = 'mean,std,rms,max,energy'
                    pg_grp.attrs['window_ms'] = self.piezo_window_ms
                    pg_grp.attrs['source_channel'] = self.piezo_channel + 1
                    pg_grp.attrs['source_adc_group'] = self.piezo_adc_group + 1
                    print(f"已保存压电特征: {piezo_features.shape}")

                mg = f.create_group('meta')
                mg.attrs['is_processed'] = True
                mg.attrs['source_pc_file'] = self._source_pc_path
                mg.attrs['source_ft_file'] = self._source_ft_path
                mg.attrs['time_offset'] = self.time_offset
                mg.attrs['num_valid_frames'] = len(valid_idx)
                mg.attrs['force_bias'] = self.force_bias.tolist()

                sg = f.create_group('segments')
                sg.attrs['count'] = len(self.segments)
                # 段坐标需要平移到新的时间起点
                # 保存的 force/timestamp 起点是 save_ts[0]，
                # 而段坐标 (s, e) 是相对于原始 ft_ts[0] 的
                delta = save_ts[0] - ft_ts[0]
                for i, (s, e) in enumerate(self.segments):
                    sg.attrs[f'seg_{i}_start'] = s - delta
                    sg.attrs[f'seg_{i}_end'] = e - delta

            QMessageBox.information(self, "完成",
                f"已保存 {len(valid_idx)} 帧到:\n{filepath}")
            self.statusBar().showMessage(f"已保存处理数据: {len(valid_idx)} 帧 → {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    # ──────────────────────────────────────────────
    #  读取处理后数据
    # ──────────────────────────────────────────────
    def _open_processed_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择处理后 HDF5 文件", self._default_dir(),
            "HDF5 文件 (*.h5 *.hdf5);;所有文件 (*.*)")
        if not filepath:
            return
        self._update_file_list(filepath)
        self._load_processed_file(filepath)

    def _load_processed_file(self, filepath):
        """加载处理后文件，恢复段信息到编辑器以支持再编辑"""
        self.txt_info.clear()
        self.h5_data = {}
        self.ft_data = {}
        self.segments = []
        self._refresh_segment_list()
        self.spin_offset.setValue(0.0)
        self._reset_bias()
        self._is_from_processed = True
        lines = []

        try:
            with h5py.File(filepath, 'r') as f:
                # 检查是否为处理后文件
                is_processed = False
                if 'meta' in f:
                    is_processed = bool(f['meta'].attrs.get('is_processed', False))

                lines += ["=" * 50,
                          "  处理后数据文件" if is_processed else "  数据文件",
                          "=" * 50, f"路径: {filepath}"]

                if is_processed:
                    meta = f['meta']
                    src_pc = meta.attrs.get('source_pc_file', '')
                    src_ft = meta.attrs.get('source_ft_file', '')
                    if isinstance(src_pc, bytes):
                        src_pc = src_pc.decode('utf-8')
                    if isinstance(src_ft, bytes):
                        src_ft = src_ft.decode('utf-8')
                    self._source_pc_path = src_pc
                    self._source_ft_path = src_ft
                    lines.append(f"源点云文件: {src_pc or 'N/A'}")
                    lines.append(f"源力觉文件: {src_ft or 'N/A'}")
                    lines.append(f"时间偏移: {meta.attrs.get('time_offset', 0):.3f} s")
                    lines.append(f"有效帧数: {meta.attrs.get('num_valid_frames', 'N/A')}")

                # 加载点云数据
                if 'vision/timestamp' in f:
                    self.h5_data['timestamp'] = f['vision/timestamp'][:]
                if 'vision/xyz' in f:
                    self.h5_data['xyz'] = f['vision/xyz'][:]
                if 'vision/dxyz' in f:
                    self.h5_data['dxyz'] = f['vision/dxyz'][:]
                if 'vision/point_id' in f:
                    self.h5_data['point_id'] = f['vision/point_id'][:]
                if 'vision/abnormal' in f:
                    self.h5_data['abnormal'] = f['vision/abnormal'][:]
                if 'reference/xyz_ref' in f:
                    self.h5_data['xyz_ref'] = f['reference/xyz_ref'][:]

                # 加载力觉数据
                if 'force' in f and 'timestamp' in f['force'] and 'values' in f['force']:
                    self.ft_data['timestamp'] = f['force/timestamp'][:]
                    self.ft_data['values'] = f['force/values'][:]
                    if 'segment_id' in f['force']:
                        self.ft_data['segment_id'] = f['force/segment_id'][:]
                    cols = f['force'].attrs.get('columns', 'fx,fy,fz,mx,my,mz')
                    if isinstance(cols, bytes):
                        cols = cols.decode('utf-8')
                    self.ft_data['columns'] = cols.split(',')

                # 数据摘要
                if 'xyz' in self.h5_data:
                    xyz = self.h5_data['xyz']
                    lines.append(f"\n点云: {xyz.shape[0]} 帧, {xyz.shape[1]} 点")
                if self.ft_data:
                    ft_ts = self.ft_data['timestamp']
                    lines.append(f"力觉: {len(ft_ts)} 采样, "
                                 f"通道: {self.ft_data['columns']}")
                    if len(ft_ts) > 1:
                        lines.append(f"时长: {ft_ts[-1] - ft_ts[0]:.2f} s")

                # 恢复段信息到编辑器（支持再编辑）
                if 'segments' in f:
                    sg = f['segments']
                    count = int(sg.attrs.get('count', 0))
                    if count > 0:
                        lines.append(f"\n有效段 ({count} 段，已恢复到编辑器):")
                        for i in range(count):
                            s = float(sg.attrs.get(f'seg_{i}_start', 0))
                            e = float(sg.attrs.get(f'seg_{i}_end', 0))
                            self.segments.append([s, e])
                            lines.append(f"  段 {i+1}: {s:.3f}s ~ {e:.3f}s")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取文件失败: {e}")
            return

        self.txt_info.setPlainText("\n".join(lines))
        self._refresh_segment_list()
        self._source_pc_path = self._source_pc_path or filepath
        self.lbl_file.setText(filepath)
        self.lbl_force_file.setText("(处理后文件，力觉数据已内嵌)")

        # 尝试从源点云文件重新加载压电原始数据
        if self._source_pc_path and os.path.isfile(self._source_pc_path):
            self._load_piezo_from_pc_file(self._source_pc_path)

        # 更新帧控件
        if 'xyz' in self.h5_data:
            self.total_frames = self.h5_data['xyz'].shape[0]
            self.spin_frame.setRange(0, self.total_frames - 1)
            self.slider_frame.setRange(0, self.total_frames - 1)
            self.spin_frame.setValue(0)
            self.on_frame_changed(0)

        # 自动打开信息面板
        self.dock_info.show()
        self.statusBar().showMessage(
            f"已加载处理文件: {os.path.basename(filepath)} | {self.total_frames} 帧")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HDF5Viewer()
    window.show()
    sys.exit(app.exec_())
