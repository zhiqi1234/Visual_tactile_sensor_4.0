import sys
import cv2
import os
import datetime
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --- 配置部分 ---
CAMERA_INDEX = 2 
SAVE_DIR_LEFT = 'calibration_left'
SAVE_DIR_RIGHT = 'calibration_right'

class VideoThread(QThread):
    """
    独立线程用于读取摄像头数据，防止卡死 GUI
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # ---------------------------------------------------------
                # 修改点：默认不旋转摄像头图像
                # 如需旋转180度，取消下面的注释
                # ---------------------------------------------------------

                # cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
                
                self.change_pixmap_signal.emit(cv_img)
            else:
                break
        
        cap.release()

    def stop(self):
        """停止线程"""
        self._run_flag = False
        self.wait()
                                                                                 
class App(QWidget):
    def __init__(self):
        super().__init__()
        
        # 修改点：更新标题以匹配当前逻辑
        self.setWindowTitle("双目镜面标定采集工具")
        self.resize(1920, 1080)
        self.disply_width = 1920
        self.display_height = 1080
        
        # 初始化当前帧变量
        self.current_frame = None

        # 创建界面布局
        self.create_ui()
        
        # 创建并启动摄像头线程
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # 检查并创建保存目录
        self.check_directories()

    def check_directories(self):
        """确保保存目录存在"""
        if not os.path.exists(SAVE_DIR_LEFT):
            os.makedirs(SAVE_DIR_LEFT)
        if not os.path.exists(SAVE_DIR_RIGHT):
            os.makedirs(SAVE_DIR_RIGHT)

    def create_ui(self):
        """创建 GUI 控件"""
        # 图像显示标签
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #333; color: white; font-size: 20px;")
        self.image_label.setText("正在连接摄像头...")

        # 采集按钮
        self.btn_capture = QPushButton("📸 采集并分割保存 (Capture)", self)
        self.btn_capture.setMinimumHeight(50)
        self.btn_capture.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_capture.clicked.connect(self.capture_and_save)

        # 状态显示
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 布局管理
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.status_label)
        vbox.addWidget(self.btn_capture)
        
        self.setLayout(vbox)

    def update_image(self, cv_img):
        """接收线程发来的图像并更新界面"""
        self.current_frame = cv_img # 这里的 cv_img 已经是旋转过的了
        
        # 将 OpenCV 的 BGR 格式转换为 Qt 的 RGB 格式
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """将 CV2 图像转换为 QPixmap 用于显示"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放以适应界面显示，保持比例
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def capture_and_save(self):
        """核心逻辑：裁剪并保存"""
        if self.current_frame is None:
            self.status_label.setText("错误：没有获取到图像")
            return

        try:
            # 1. 获取图像尺寸
            height, width, _ = self.current_frame.shape
            
            # 2. 计算中点
            mid_point = width // 2

            # 3. 分割图像 (Numpy 切片)
            crop_left = self.current_frame[:, :mid_point]
            crop_right = self.current_frame[:, mid_point:]

            # -------------------------------------------------------------
            # 修改点：补全空白逻辑
            # 要求：左侧裁切就对右侧补上空白，右侧裁切就对左侧补上空白
            # -------------------------------------------------------------
            
            # 定义空白颜色：[0, 0, 0] 为黑色，[255, 255, 255] 为白色
            pad_color = [0, 0, 0] 

            # 左图处理：保留左边内容，在右边填充空白
            # copyMakeBorder 参数: src, top, bottom, left, right, borderType, value
            img_left = cv2.copyMakeBorder(
                crop_left, 
                0, 0, 0, width - mid_point,  # 上下左不补，右边补足宽度
                cv2.BORDER_CONSTANT, 
                value=pad_color
            )

            # 右图处理：保留右边内容，在左边填充空白
            img_right = cv2.copyMakeBorder(
                crop_right, 
                0, 0, mid_point, 0,          # 上下右不补，左边补足宽度
                cv2.BORDER_CONSTANT, 
                value=pad_color
            )

            # 4. 生成文件名 (使用时间戳防止重名)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"img_{timestamp}.jpg"

            # 5. 拼接完整路径
            path_left = os.path.join(SAVE_DIR_LEFT, filename)
            path_right = os.path.join(SAVE_DIR_RIGHT, filename)

            # 6. 保存图片 (保存处理后的补全图)
            cv2.imwrite(path_left, img_left)
            cv2.imwrite(path_right, img_right)

            # 7. 更新状态
            self.status_label.setText(f"已保存: {filename} \n(分辨率: {width}x{height} -> 拆分并补全)")
            print(f"Saved {path_left} and {path_right}")

        except Exception as e:
            self.status_label.setText(f"保存失败: {e}")

    def closeEvent(self, event):
        """关闭窗口时清理线程"""
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())