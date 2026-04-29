import sys
import serial
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QLabel, QGroupBox)
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
import pyqtgraph as pg
import serial.tools.list_ports


class TactileFinger(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("压电触觉传感器数据采集")
        self.resize(1200, 800)

        # 创建界面
        self.setup_ui()

        # 串口接收线程
        self.serial_receiver = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(200)

        # 数据存储
        self.data_ptac_adc1 = []
        self.data_ptac_adc2 = []
        self.data_ptac_adc3 = []
        self.data_ptac_adc4 = []
        self.data_ptac_adc5 = []

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_panel)

        # 串口选择
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("串口:"))
        self.cbb_ptac_port = QComboBox()
        self.cbb_ptac_port.addItems(self.get_serial_ports())
        port_layout.addWidget(self.cbb_ptac_port)
        control_layout.addLayout(port_layout)

        # 波特率选择
        baud_layout = QHBoxLayout()
        baud_layout.addWidget(QLabel("波特率:"))
        self.cbb_ptac_baud = QComboBox()
        self.cbb_ptac_baud.addItems(["921600", "115200"])
        baud_layout.addWidget(self.cbb_ptac_baud)
        control_layout.addLayout(baud_layout)

        # ADC组选择
        adc_layout = QHBoxLayout()
        adc_layout.addWidget(QLabel("ADC组:"))
        self.cbb_cs = QComboBox()
        self.cbb_cs.addItems(["1", "2", "3", "4", "5"])
        self.cbb_cs.currentIndexChanged.connect(self.update_plot)
        adc_layout.addWidget(self.cbb_cs)
        control_layout.addLayout(adc_layout)

        # 连接/断开按钮
        self.btn_ptac_connect = QPushButton("连接")
        self.btn_ptac_connect.clicked.connect(self.start_reading)
        control_layout.addWidget(self.btn_ptac_connect)

        self.btn_ptac_disconnect = QPushButton("断开")
        self.btn_ptac_disconnect.clicked.connect(self.stop_reading)
        control_layout.addWidget(self.btn_ptac_disconnect)

        # 刷新串口按钮
        self.btn_refresh = QPushButton("刷新串口")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.btn_refresh)

        control_layout.addStretch()
        control_panel.setFixedWidth(250)
        main_layout.addWidget(control_panel)

        # 右侧绘图区域
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')

        self.plots = []
        self.curves = []
        for i in range(8):
            plot_item = self.plot_widget.addPlot(row=i // 2, col=i % 2, title=f"CH:{i + 1}")
            plot_item.setYRange(-3.3, 3.3)
            plot_item.setTitle(f"CH:{i + 1}", color='k')
            plot_item.getAxis('left').setPen(pg.mkPen(color='k'))
            plot_item.getAxis('bottom').setPen(pg.mkPen(color='k'))
            plot_item.getAxis('left').setTextPen(pg.mkPen(color='k'))
            plot_item.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
            self.plots.append(plot_item)
            curve = plot_item.plot(pen='b')
            self.curves.append(curve)

        main_layout.addWidget(self.plot_widget)

    def get_serial_ports(self):
        """获取可用的串口端口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def refresh_ports(self):
        """刷新串口列表"""
        self.cbb_ptac_port.clear()
        self.cbb_ptac_port.addItems(self.get_serial_ports())

    def update_data(self, data_frame_datas):
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

    def update_plot(self):
        group_index = self.cbb_cs.currentIndex()
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

        x_data = list(range(max(0, len(data_adc_i) - 2000), len(data_adc_i)))

        for i, curve in enumerate(self.curves):
            y_data = [row[i] for row in data_adc_i[-2000:] if len(row) > i]
            curve.setData(x_data, y_data)

    def bytes_to_decimal(self, data):
        byte1, byte2, byte3 = data
        adc_value = (byte1 << 16) | (byte2 << 8) | byte3
        if adc_value & 0x800000:
            adc_value -= 0x1000000
        return adc_value

    def start_reading(self):
        """点击连接按钮后启动串口通信线程"""
        port = self.cbb_ptac_port.currentText()
        if not port:
            print("请选择串口")
            return
        baudrate = int(self.cbb_ptac_baud.currentText())
        self.data_ptac_adc1 = []
        self.data_ptac_adc2 = []
        self.data_ptac_adc3 = []
        self.data_ptac_adc4 = []
        self.data_ptac_adc5 = []
        if self.serial_receiver is None or not self.serial_receiver.isRunning():
            self.serial_receiver = SerialReceiver(port, baudrate)
            self.serial_receiver.data_received.connect(self.update_data)
            self.serial_receiver.start()
            print(f"已连接到 {port}，正在清空缓冲区...")

    def stop_reading(self):
        if self.serial_receiver and self.serial_receiver.isRunning():
            self.serial_receiver.stop()
            print("已断开连接")

    def closeEvent(self, event):
        if self.serial_receiver:
            self.serial_receiver.stop()
        event.accept()


class SerialReceiver(QThread):
    data_received = pyqtSignal(bytes)

    def __init__(self, port='COM32', baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.buffer = b''

    def run(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial.reset_input_buffer()  # 清空串口接收缓冲区，丢弃积压的旧数据
            self.running = True
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = TactileFinger()
    w.show()
    sys.exit(app.exec())
