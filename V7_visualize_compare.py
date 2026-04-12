# -*- coding: utf-8 -*-
'''
V7 HDF5数据可视化 - 预测力与实际力对比
功能：
1. 读取V7生成的HDF5文件
2. 6个横向坐标轴显示预测力vs实际力对比
3. 点云显示（参照V5方式）
4. 预测力方向和大小用箭头显示在点云上
'''
import sys
import os
import numpy as np
import h5py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QSlider,
                             QLabel, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class HDF5Visualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.h5_file = None
        self.vision_data = None
        self.force_data = None
        self.xyz_ref = None
        self.current_idx = 0
        self.playing = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('V7 HDF5 可视化 - 预测力vs实际力对比')
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton('加载HDF5文件')
        self.btn_load.clicked.connect(self.load_h5_file)
        self.btn_play = QPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 滑块
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setEnabled(False)
        self.label_frame = QLabel('帧: 0/0')
        slider_layout.addWidget(QLabel('进度:'))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label_frame)
        main_layout.addLayout(slider_layout)

        # 分割器：上方6个力对比图，下方点云
        splitter = QSplitter(Qt.Vertical)

        # 上方：6个力对比图
        self.fig_force, self.axes_force = plt.subplots(2, 3, figsize=(15, 6))
        self.canvas_force = FigureCanvas(self.fig_force)
        splitter.addWidget(self.canvas_force)

        # 下方：点云
        self.fig_cloud = plt.figure(figsize=(8, 6))
        self.ax_cloud = self.fig_cloud.add_subplot(111, projection='3d')
        self.canvas_cloud = FigureCanvas(self.fig_cloud)
        cloud_widget = QWidget()
        cloud_layout = QVBoxLayout(cloud_widget)
        cloud_layout.addWidget(NavigationToolbar(self.canvas_cloud, self))
        cloud_layout.addWidget(self.canvas_cloud)
        splitter.addWidget(cloud_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_h5_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择HDF5文件', '', 'HDF5 Files (*.h5 *.hdf5)')
        if not file_path:
            return

        try:
            if self.h5_file:
                self.h5_file.close()

            self.h5_file = h5py.File(file_path, 'r')

            # 读取数据
            self.vision_data = {
                'timestamp': self.h5_file['/vision/timestamp'][:],
                'xyz': self.h5_file['/vision/xyz'][:],
                'dxyz': self.h5_file['/vision/dxyz'][:],
                'predicted_force': self.h5_file['/vision/predicted_force'][:]
            }

            self.force_data = {
                'timestamp': self.h5_file['/force/timestamp'][:],
                'values': self.h5_file['/force/values'][:]
            }

            self.xyz_ref = self.h5_file['/reference/xyz_ref'][:]

            # 同步力数据到视觉帧
            self.sync_force_to_vision()

            # 初始化UI
            n_frames = len(self.vision_data['timestamp'])
            self.slider.setMaximum(n_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.current_idx = 0

            self.update_display()
            self.setWindowTitle(f'V7 HDF5 可视化 - {os.path.basename(file_path)}')

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, '错误', f'加载文件失败：{str(e)}')

    def sync_force_to_vision(self):
        """将力数据同步到视觉帧时间戳"""
        vision_ts = self.vision_data['timestamp']
        force_ts = self.force_data['timestamp']
        force_vals = self.force_data['values']

        synced_force = np.zeros((len(vision_ts), 6))
        for i, vts in enumerate(vision_ts):
            idx = np.argmin(np.abs(force_ts - vts))
            synced_force[i] = force_vals[idx]

        self.vision_data['synced_force'] = synced_force

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText('暂停')
            self.timer.start(33)  # ~30fps
        else:
            self.btn_play.setText('播放')
            self.timer.stop()

    def next_frame(self):
        if self.current_idx < self.slider.maximum():
            self.current_idx += 1
            self.slider.setValue(self.current_idx)
        else:
            self.toggle_play()

    def on_slider_changed(self, value):
        self.current_idx = value
        self.update_display()

    def update_display(self):
        if self.vision_data is None:
            return

        idx = self.current_idx
        n_frames = len(self.vision_data['timestamp'])
        self.label_frame.setText(f'帧: {idx+1}/{n_frames}')

        # 更新力对比图
        self.update_force_comparison(idx)

        # 更新点云
        self.update_point_cloud(idx)

    def update_force_comparison(self, idx):
        """更新6个力分量对比图"""
        predicted = self.vision_data['predicted_force'][idx]
        actual = self.vision_data['synced_force'][idx]

        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        n_frames = len(self.vision_data['timestamp'])

        # 一分钟窗口（假设30fps，1800帧）
        window_size = 1800
        start_idx = max(0, idx - window_size // 2)
        end_idx = min(n_frames, start_idx + window_size)
        start_idx = max(0, end_idx - window_size)

        for i, ax in enumerate(self.axes_force.flat):
            ax.clear()

            # 只绘制窗口内的数据
            x_range = np.arange(start_idx, end_idx)
            pred_window = self.vision_data['predicted_force'][start_idx:end_idx, i]
            actual_window = self.vision_data['synced_force'][start_idx:end_idx, i]

            ax.plot(x_range, pred_window, 'b-', label='预测', linewidth=1.5, alpha=0.7)
            ax.plot(x_range, actual_window, 'r-', label='实际', linewidth=1.5, alpha=0.7)
            ax.axvline(idx, color='green', linestyle='--', linewidth=2, alpha=0.8)

            ax.set_title(f'{labels[i]}: 预测={predicted[i]:.2f}, 实际={actual[i]:.2f}', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('帧', fontsize=8)
            ax.set_xlim([start_idx, end_idx])

        self.fig_force.tight_layout()
        self.canvas_force.draw()

    def update_point_cloud(self, idx):
        """更新点云显示"""
        self.ax_cloud.clear()

        xyz = self.vision_data['xyz'][idx]
        predicted_force = self.vision_data['predicted_force'][idx]

        # 绘制参考点云（灰色）
        if self.xyz_ref is not None and len(self.xyz_ref) > 0:
            self.ax_cloud.scatter(self.xyz_ref[:, 0], self.xyz_ref[:, 1], self.xyz_ref[:, 2],
                                 c='gray', s=1, alpha=0.3, label='参考')

        # 绘制当前点云（蓝色）
        if len(xyz) > 0:
            self.ax_cloud.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                 c='blue', s=10, alpha=0.8, label='当前')

        # 绘制预测力箭头
        if len(xyz) > 0:
            center = np.mean(xyz, axis=0)
            force_vec = predicted_force[:3]  # Fx, Fy, Fz
            force_mag = np.linalg.norm(force_vec)

            if force_mag > 0.5:
                force_dir = force_vec / force_mag
                arrow_len = force_mag * 2.0

                self.ax_cloud.quiver(center[0], center[1], center[2],
                                    force_dir[0] * arrow_len, force_dir[1] * arrow_len, force_dir[2] * arrow_len,
                                    color='red', arrow_length_ratio=0.2,
                                    linewidth=3, label=f'预测力 ({force_mag:.2f}N)')

        self.ax_cloud.set_xlabel('X')
        self.ax_cloud.set_ylabel('Y')
        self.ax_cloud.set_zlabel('Z')
        self.ax_cloud.legend()
        self.ax_cloud.set_title(f'点云 + 预测力方向 (帧 {idx+1})')

        # 设置坐标轴范围
        if self.xyz_ref is not None and len(self.xyz_ref) > 0:
            all_points = np.vstack([self.xyz_ref, xyz]) if len(xyz) > 0 else self.xyz_ref
        elif len(xyz) > 0:
            all_points = xyz
        else:
            all_points = np.array([[0, 0, 0]])

        margin = 5
        self.ax_cloud.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
        self.ax_cloud.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
        self.ax_cloud.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])

        self.canvas_cloud.draw()

    def closeEvent(self, event):
        if self.h5_file:
            self.h5_file.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HDF5Visualizer()
    window.show()
    sys.exit(app.exec_())
