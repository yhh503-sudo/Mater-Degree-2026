import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QHBoxLayout, 
                             QVBoxLayout, QGridLayout, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class UltrasoundApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Analysis(STD, Histogram)")
        self.setGeometry(100, 100, 1000, 600)
        
        self.cv_img = None
        self.roi_img = None
        
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        
        # --- 상단 컨트롤 패널 ---
        control_panel = QHBoxLayout()
        
        # 1. 파일 열기
        self.btn_open = QPushButton("Open Image")
        self.btn_open.setHeight = 40
        self.btn_open.clicked.connect(self.load_image)
        self.lbl_path = QLabel("Loaded: None")
        self.lbl_path.setStyleSheet("color: green; background-color: #E0E0E0; padding: 5px;")
        
        # 2. ROI 좌표 설정 (SpinBoxes)
        grid_roi = QGridLayout()
        grid_roi.addWidget(QLabel("Start X"), 0, 0)
        self.spn_x = QSpinBox()
        self.spn_x.setRange(0, 5000)
        self.spn_x.setValue(13)
        grid_roi.addWidget(self.spn_x, 0, 1)
        
        grid_roi.addWidget(QLabel("Start Y"), 0, 2)
        self.spn_y = QSpinBox()
        self.spn_y.setRange(0, 5000)
        self.spn_y.setValue(13)
        grid_roi.addWidget(self.spn_y, 0, 3)

        grid_roi.addWidget(QLabel("X Leng"), 1, 0)
        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 5000)
        self.spn_w.setValue(50)
        grid_roi.addWidget(self.spn_w, 1, 1)

        grid_roi.addWidget(QLabel("Y Leng"), 1, 2)
        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 5000)
        self.spn_h.setValue(50)
        grid_roi.addWidget(self.spn_h, 1, 3)

        # 3. 버튼들
        self.btn_roi_set = QPushButton("ROI SET")
        self.btn_roi_set.clicked.connect(self.set_roi)
        
        btn_vbox = QVBoxLayout()
        self.btn_std = QPushButton("STD")
        self.btn_std.clicked.connect(self.calc_std)
        self.btn_hist = QPushButton("Grey\nDistribution")
        self.btn_hist.clicked.connect(self.plot_histogram)
        btn_vbox.addWidget(self.btn_std)
        btn_vbox.addWidget(self.btn_hist)

        # 레이아웃 조합
        control_panel.addWidget(self.btn_open)
        control_panel.addWidget(self.lbl_path)
        control_panel.addLayout(grid_roi)
        control_panel.addWidget(self.btn_roi_set)
        control_panel.addLayout(btn_vbox)
        
        main_layout.addLayout(control_panel)

        # --- 하단 디스플레이 영역 (이미지 + 그래프/결과) ---
        display_layout = QHBoxLayout()
        
        # 좌측: 이미지 표시 라벨
        self.lbl_image = QLabel("Image Area")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 1px solid gray;")
        
        # 우측: Matplotlib 히스토그램 & STD 결과 표시 레이아웃
        right_vbox = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.lbl_std_result = QLabel("STD = -")
        self.lbl_std_result.setAlignment(Qt.AlignCenter)
        self.lbl_std_result.setStyleSheet("background-color: yellow; font-size: 16px; font-weight: bold; border: 1px solid orange; padding: 5px;")

        right_vbox.addWidget(self.canvas)
        right_vbox.addWidget(self.lbl_std_result)

        display_layout.addWidget(self.lbl_image, 1)
        display_layout.addLayout(right_vbox, 1)

        main_layout.addLayout(display_layout)

        # ★ 아래 한 줄을 맨 끝에 추가하세요!
        main_widget.setLayout(main_layout)
        
        # 컨트롤 패널 디자인 다듬기
        main_widget.setStyleSheet("""
            QWidget { font-size: 12px; }
            QPushButton { background-color: #D3D3D3; border: 1px solid #999; padding: 5px; min-width: 60px; }
            QPushButton:hover { background-color: #C0C0C0; }
        """)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Image Files (*.jpg *.png *.bmp)')
        if fname:
            # 한글 경로 지원을 위한 cv2 imread 처리
            img_array = np.fromfile(fname, np.uint8)
            self.cv_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            file_name = fname.split('/')[-1]
            self.lbl_path.setText(f"Loaded:{file_name}")
            self.display_image(self.cv_img)

    def display_image(self, img):
        if img is None:
            return
        
        # BGR -> RGB 변환 (흑백도 3채널로 맞춤)
        if len(img.shape) == 2:
            display_cv = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        h, w, ch = display_cv.shape
        bytes_per_line = ch * w
        q_img = QImage(display_cv.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Label 크기에 맞춰 비율 유지하며 축소/확대
        scaled_pixmap = pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio)
        self.lbl_image.setPixmap(scaled_pixmap)

    def set_roi(self):
        if self.cv_img is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        x = self.spn_x.value()
        y = self.spn_y.value()
        w = self.spn_w.value()
        h = self.spn_h.value()

        img_h, img_w = self.cv_img.shape[:2]
        
        # ROI 범위가 이미지를 벗어나지 않도록 클리핑
        x_end = min(x + w, img_w)
        y_end = min(y + h, img_h)

        # 실제 연산용 ROI 추출
        self.roi_img = self.cv_img[y:y_end, x:x_end]

        # 화면 표시용 복사본 이미지 생성 및 ROI 박스(파란색) 그리기
        img_with_box = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 150, 0), 2)  # 하늘색/파란색 상자
        
        self.display_image(img_with_box)

    def calc_std(self):
        if self.roi_img is None or self.roi_img.size == 0:
            QMessageBox.warning(self, "Warning", "Set ROI first!")
            return

        # ROI 영역 내 명암값의 표준편차 계산
        std_val = np.std(self.roi_img)
        self.lbl_std_result.setText(f"STD = {std_val:.2f}")

    def plot_histogram(self):
        if self.roi_img is None or self.roi_img.size == 0:
            QMessageBox.warning(self, "Warning", "Set ROI first!")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # OpenCv 히스토그램 계산 (0~255 픽셀 분포)
        hist = cv2.calcHist([self.roi_img], [0], None, [256], [0, 256])

        ax.plot(hist, color='blue', linewidth=1)
        ax.set_xlim([0, 260])
        max_val = np.max(hist)
        ax.set_ylim([0, max_val if max_val > 0 else 100])
        
        # 그래프 여백 정리
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UltrasoundApp()
    ex.show()
    sys.exit(app.exec_())