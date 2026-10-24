import sys
import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis
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
        self.setWindowTitle("Multi-Image ROI Feature & Histogram Analysis")
        self.setGeometry(50, 50, 1500, 800)  # 전체 윈도우 크기 확대
        
        self.file_paths = []
        self.cv_img_first = None
        
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        
        # --- 상단 컨트롤 패널 ---
        control_panel = QHBoxLayout()
        
        # 1. 여러 파일 열기
        self.btn_open = QPushButton("Open Images")
        self.btn_open.clicked.connect(self.load_images)
        self.lbl_path = QLabel("Loaded: 0 files")
        self.lbl_path.setStyleSheet("color: green; background-color: #E0E0E0; padding: 5px;")
        
        # 2. ROI 좌표 설정 (SpinBoxes)
        grid_roi = QGridLayout()
        grid_roi.addWidget(QLabel("Start X"), 0, 0)
        self.spn_x = QSpinBox()
        self.spn_x.setRange(0, 5000)
        self.spn_x.setValue(0)
        grid_roi.addWidget(self.spn_x, 0, 1)
        
        grid_roi.addWidget(QLabel("Start Y"), 0, 2)
        self.spn_y = QSpinBox()
        self.spn_y.setRange(0, 5000)
        self.spn_y.setValue(330)
        grid_roi.addWidget(self.spn_y, 0, 3)

        grid_roi.addWidget(QLabel("X Leng"), 1, 0)
        self.spn_w = QSpinBox()
        self.spn_w.setRange(1, 5000)
        self.spn_w.setValue(420)
        grid_roi.addWidget(self.spn_w, 1, 1)

        grid_roi.addWidget(QLabel("Y Leng"), 1, 2)
        self.spn_h = QSpinBox()
        self.spn_h.setRange(1, 5000)
        self.spn_h.setValue(40)
        grid_roi.addWidget(self.spn_h, 1, 3)

        # 3. 분석 실행 버튼
        self.btn_roi_set = QPushButton("ROI Preview")
        self.btn_roi_set.clicked.connect(self.preview_roi)
        
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_analyze.clicked.connect(self.run_analysis)

        # 레이아웃 조합
        control_panel.addWidget(self.btn_open)
        control_panel.addWidget(self.lbl_path)
        control_panel.addLayout(grid_roi)
        control_panel.addWidget(self.btn_roi_set)
        control_panel.addWidget(self.btn_analyze)
        
        main_layout.addLayout(control_panel)

        # --- 하단 디스플레이 영역 ---
        display_layout = QHBoxLayout()
        
        # 좌측: 첫번째 이미지 및 ROI 설정 표시
        self.lbl_image = QLabel("First Image (ROI Reference)")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 1px solid gray;")
        
        # 우측: Matplotlib 그래프 2개 (히스토그램, Feature 추세)
        right_vbox = QVBoxLayout()
        self.figure, (self.ax_hist, self.ax_feat) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        
        right_vbox.addWidget(self.canvas)

        display_layout.addWidget(self.lbl_image, 1)
        display_layout.addLayout(right_vbox, 2)

        main_layout.addLayout(display_layout)
        main_widget.setLayout(main_layout)
        
        main_widget.setStyleSheet("""
            QWidget { font-size: 12px; }
            QPushButton { background-color: #D3D3D3; border: 1px solid #999; padding: 6px; min-width: 70px; }
            QPushButton:hover { background-color: #C0C0C0; }
        """)

    def load_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open Images', '', 'Image Files (*.jpg *.png *.bmp)')
        if fnames:
            self.file_paths = fnames
            self.lbl_path.setText(f"Loaded: {len(fnames)} files")
            
            # 첫 번째 이미지를 대표 이미지로 로드
            img_array = np.fromfile(self.file_paths[0], np.uint8)
            self.cv_img_first = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.cv_img_first)

    def display_image(self, img):
        if img is None:
            return
        
        if len(img.shape) == 2:
            display_cv = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        h, w, ch = display_cv.shape
        bytes_per_line = ch * w
        q_img = QImage(display_cv.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio)
        self.lbl_image.setPixmap(scaled_pixmap)

    def preview_roi(self):
        if self.cv_img_first is None:
            QMessageBox.warning(self, "Warning", "Please load images first!")
            return

        x = self.spn_x.value()
        y = self.spn_y.value()
        w = self.spn_w.value()
        h = self.spn_h.value()

        img_with_box = cv2.cvtColor(self.cv_img_first, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 150, 0), 2)
        self.display_image(img_with_box)

    def calculate_features(self, roi):
        flat_roi = roi.ravel()
        
        mean_val = np.mean(flat_roi)
        std_val = np.std(flat_roi)
        skew_val = skew(flat_roi)
        kurt_val = kurtosis(flat_roi)
        
        # Entropy 계산
        hist, _ = np.histogram(flat_roi, bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy_val = -np.sum(hist_norm * np.log2(hist_norm))
        
        return {
            'Mean': mean_val,
            'STD': std_val,
            'Skewness': skew_val,
            'Kurtosis': kurt_val,
            'Entropy': entropy_val
        }

    def run_analysis(self):
        if not self.file_paths:
            QMessageBox.warning(self, "Warning", "Please load images first!")
            return

        x = self.spn_x.value()
        y = self.spn_y.value()
        w = self.spn_w.value()
        h = self.spn_h.value()

        labels = []
        hist_list = []
        features_list = {
            'Mean': [],
            'STD': [],
            'Skewness': [],
            'Kurtosis': [],
            'Entropy': []
        }

        # 이미지별 처리 진행
        for idx, path in enumerate(self.file_paths):
            file_name = os.path.basename(path)
            labels.append(f"Img_{idx+1}")  # 간결한 X축 범주명 (또는 file_name)

            img_array = np.fromfile(path, np.uint8)
            cv_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            img_h, img_w = cv_img.shape[:2]
            x_end = min(x + w, img_w)
            y_end = min(y + h, img_h)
            
            roi = cv_img[y:y_end, x:x_end]

            # 1. 히스토그램 계산
            hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).ravel()
            hist_list.append((file_name, hist))

            # 2. Feature 계산
            feats = self.calculate_features(roi)
            for k, v in feats.items():
                features_list[k].append(v)

        # --- 그래프 서식 설정 및 그리기 ---
        self.ax_hist.clear()
        self.ax_feat.clear()

        # 그래프 1: 히스토그램 겹침 (Overlapped Histograms)
        colors = plt.cm.jet(np.linspace(0, 1, len(hist_list)))
        for i, (fname, hist) in enumerate(hist_list):
            self.ax_hist.plot(hist, color=colors[i], alpha=0.7, label=labels[i])
            
        self.ax_hist.set_title("Overlapped Histograms")
        self.ax_hist.set_xlabel("Pixel Intensity")
        self.ax_hist.set_ylabel("Pixel Count")
        self.ax_hist.set_xlim([0, 256])
        if len(hist_list) <= 10:
            self.ax_hist.legend(loc='upper right', fontsize='small')

        # 그래프 2: B-Scan별 Feature 추세
        x_indices = range(len(labels))
        for feat_name, values in features_list.items():
            self.ax_feat.plot(x_indices, values, marker='o', label=feat_name)

        self.ax_feat.set_title("Features per B-Scan Image")
        self.ax_feat.set_xlabel("B-Scan Image Index")
        self.ax_feat.set_ylabel("Feature Value")
        self.ax_feat.set_xticks(x_indices)
        self.ax_feat.set_xticklabels(labels, rotation=45, ha='right')
        self.ax_feat.grid(True, linestyle='--', alpha=0.5)
        self.ax_feat.legend(loc='upper right', fontsize='small')

        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UltrasoundApp()
    ex.show()
    sys.exit(app.exec_())