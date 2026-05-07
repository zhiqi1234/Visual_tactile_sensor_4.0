# -*- coding: utf-8 -*-
import sys
import os
import cv2
import time
import datetime
import threading
import queue
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGroupBox, QStatusBar, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# --------------------------
# 1. 图像保存线程 (消费者)
# --------------------------
class ImageSaver(QThread):
    """
    专门用于保存图像的线程，避免IO操作阻塞摄像头采集
    """
    update_log = pyqtSignal(str)  # 用于发送日志信号

    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.save_dir = None
        self.frame_count = 0

    def set_save_directory(self, base_name="Vptac_shape_data"):
        # 获取当前时间
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{base_name}_{now_str}"
        
        # 创建文件夹
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        self.save_dir = folder_name
        self.frame_count = 0
        return self.save_dir

    def add_image(self, image):
        if self.running and self.save_dir:
            self.queue.put(image)

    def run(self):
        while self.running:
            try:
                # 从队列获取图像，超时1秒避免死锁
                img = self.queue.get(timeout=1)
                
                self.frame_count += 1
                filename = f"{self.frame_count:03d}.png"
                file_path = os.path.join(self.save_dir, filename)
                
                # 写入 PNG (压缩等级默认，无损)
                cv2.imwrite(file_path, img)
                
                # 可选：每保存10帧打印一次日志，避免刷屏
                if self.frame_count % 10 == 0:
                    self.update_log.emit(f"已保存: {filename}")
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.update_log.emit(f"保存出错: {e}")

    def stop(self):
        self.running = False
        self.wait()

# --------------------------
# 2. 摄像头采集线程 (生产者)
# --------------------------
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, fps=24):
        super().__init__()
        self.running = True
        self.recording = False
        self.target_fps = fps
        self.video_cap = None

    def run(self):
        # 打开默认摄像头 (索引0)
        self.video_cap = cv2.VideoCapture(2)
        
        # 尝试设置硬件参数
        self.video_cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 如需高清可解注
        # self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 计算帧间隔 (秒)
        frame_interval = 1.0 / self.target_fps

        while self.running:
            start_time = time.time()
            
            ret, frame = self.video_cap.read()
            if ret:
                # =================================================
                # 【修改处】不再旋转 180 度
                # =================================================
                # frame = cv2.rotate(frame, cv2.ROTATE_180)

                # 发送信号给GUI显示
                self.change_pixmap_signal.emit(frame)
            
            # 控制帧率
            elapsed = time.time() - start_time
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        if self.video_cap:
            self.video_cap.release()

    def start_record(self):
        self.recording = True

    def stop_record(self):
        self.recording = False

    def stop(self):
        self.running = False
        self.wait()

# --------------------------
# 3. 主界面 GUI
# --------------------------
class RecorderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Vptac 数据采集助手"
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 800)

        # 状态变量
        self.is_recording = False
        self.image_saver = ImageSaver()
        self.image_saver.update_log.connect(self.log_status)
        self.image_saver.start()

        self.camera_thread = CameraThread(fps=30)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

        self.init_ui()

    def init_ui(self):
        # 主窗口部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- 1. 图像显示区域 ---
        self.display_group = QGroupBox("实时预览")
        display_layout = QVBoxLayout()
        
        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("正在连接摄像头...")
        self.image_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        display_layout.addWidget(self.image_label)
        self.display_group.setLayout(display_layout)
        self.layout.addWidget(self.display_group)

        # --- 2. 控制面板 ---
        self.control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        self.lbl_info = QLabel("状态: 就绪")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.btn_record = QPushButton("开始录制")
        self.btn_record.setMinimumHeight(50)
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
        self.btn_record.clicked.connect(self.toggle_recording)

        control_layout.addWidget(self.lbl_info)
        control_layout.addWidget(self.btn_record)
        self.control_group.setLayout(control_layout)
        self.layout.addWidget(self.control_group)

        # --- 3. 状态栏 ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统初始化完成")

    def update_image(self, cv_img):
        """接收摄像头线程的图像信号，更新UI，并分发给保存线程"""
        # 如果正在录制，将原始图像发送给保存线程
        if self.is_recording:
            # 注意：cv_img 此时已经在 CameraThread 中被旋转过了
            self.image_saver.add_image(cv_img.copy())

        # 转换图像格式用于显示 (BGR -> RGB)
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 保持比例缩放以适应标签大小
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def toggle_recording(self):
        if not self.is_recording:
            # --- 开始录制 ---
            # 1. 设置保存路径
            save_path = self.image_saver.set_save_directory()
            
            # 2. 更新UI状态
            self.is_recording = True
            self.btn_record.setText("停止录制")
            self.btn_record.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 16px;")
            self.lbl_info.setText(f"录制中... 文件夹: {os.path.basename(save_path)}")
            self.status_bar.showMessage(f"开始录制，保存至: {save_path}")
            
            self.camera_thread.start_record()
            
        else:
            # --- 停止录制 ---
            self.is_recording = False
            self.camera_thread.stop_record()
            
            self.btn_record.setText("开始录制")
            self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
            self.lbl_info.setText(f"录制结束。共保存 {self.image_saver.frame_count} 帧。")
            self.status_bar.showMessage(f"录制已停止，数据保存在: {self.image_saver.save_dir}")
            
            QMessageBox.information(self, "录制完成", 
                                    f"已停止录制。\n\n"
                                    f"保存文件夹: {self.image_saver.save_dir}\n"
                                    f"总帧数: {self.image_saver.frame_count}")

    def log_status(self, msg):
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """关闭窗口时清理线程"""
        self.camera_thread.stop()
        self.image_saver.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecorderGUI()
    window.show()
    sys.exit(app.exec_())