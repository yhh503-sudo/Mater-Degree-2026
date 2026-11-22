import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QHBoxLayout, 
                             QVBoxLayout, QGridLayout)
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class UltrasoundApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Image ROI Feature & Histogram Analysis (with CNR)")
        self.setGeometry(50, 50, 1600, 800)
        
        self.file_paths = []
        self.cv_img_first = None       
        self.initUI()

    def create_roi_group(self, title, default_vals):
        """반복되는 ROI 입력창 구성을 위한 헬퍼 함수"""
        group_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        
        grid.addWidget(QLabel("Start X"), 0, 0)
        spn_x = QSpinBox(); spn_x.setRange(0, 5000); spn_x.setValue(default_vals[0])
        grid.addWidget(spn_x, 0, 1)
        
        grid.addWidget(QLabel("Start Y"), 0, 2)
        spn_y = QSpinBox(); spn_y.setRange(0, 5000); spn_y.setValue(default_vals[1])
        grid.addWidget(spn_y, 0, 3)

        grid.addWidget(QLabel("X Leng"), 1, 0)
        spn_w = QSpinBox(); spn_w.setRange(1, 5000); spn_w.setValue(default_vals[2])
        grid.addWidget(spn_w, 1, 1)

        grid.addWidget(QLabel("Y Leng"), 1, 2)
        spn_h = QSpinBox(); spn_h.setRange(1, 5000); spn_h.setValue(default_vals[3])
        grid.addWidget(spn_h, 1, 3)

        layout.addLayout(grid)
        group_widget.setLayout(layout)
        return group_widget, spn_x, spn_y, spn_w, spn_h

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        
        # --- 상단 컨트롤 패널 (3가지 ROI 입력란 가로 배치) ---
        control_panel = QHBoxLayout()
        
        # 1. 파일 열기 부분
        file_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Images")
        self.btn_open.clicked.connect(self.load_images)
        self.lbl_path = QLabel("Loaded: 0 files")
        self.lbl_path.setStyleSheet("color: green; background-color: #E0E0E0; padding: 5px;")
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.lbl_path)
        
        # 2. 3가지 ROI 그룹 생성
        self.grp_gen, self.gen_x, self.gen_y, self.gen_w, self.gen_h = self.create_roi_group(
            "General ROI for Mean, STD, Entropy", (0, 330, 420, 40))
            
        self.grp_cnr, self.cnr_x, self.cnr_y, self.cnr_w, self.cnr_h = self.create_roi_group(
            "Specific ROI for CNR", (0, 0, 420, 420))
            
        self.grp_noise, self.noise_x, self.noise_y, self.noise_w, self.noise_h = self.create_roi_group(
            "Specific Noise Floor for CNR", (0, 0, 420, 420))
        
        # 3. 버튼 레이아웃
        btn_layout = QVBoxLayout()
        self.btn_roi_set = QPushButton("ROI Preview")
        self.btn_roi_set.clicked.connect(self.preview_roi)
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_analyze.clicked.connect(self.run_analysis)
        btn_layout.addWidget(self.btn_roi_set)
        btn_layout.addWidget(self.btn_analyze)

        # 상단 레이아웃 조합
        control_panel.addLayout(file_layout)
        control_panel.addSpacing(20)
        control_panel.addWidget(self.grp_gen)
        control_panel.addWidget(self.grp_cnr)
        control_panel.addWidget(self.grp_noise)
        control_panel.addSpacing(20)
        control_panel.addLayout(btn_layout)
        
        main_layout.addLayout(control_panel)

        # --- 하단 디스플레이 영역 ---
        display_layout = QHBoxLayout()
        
        self.lbl_image = QLabel("First Image (ROI Reference)")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 1px solid gray;")
        
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
            QPushButton { background-color: #D3D3D3; border: 1px solid #999; padding: 6px; min-width: 90px; }
            QPushButton:hover { background-color: #C0C0C0; }
        """)

    def load_images(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open Images', '', 'Image Files (*.jpg *.png *.bmp)')
        if fnames:
            self.file_paths = fnames
            self.lbl_path.setText(f"Loaded: {len(fnames)} files")
            
            img_array = np.fromfile(self.file_paths[0], np.uint8)
            self.cv_img_first = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.cv_img_first)

    def display_image(self, img):
        if img is None: return
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

        img_with_box = cv2.cvtColor(self.cv_img_first, cv2.COLOR_GRAY2BGR)
        
        # General ROI - Blue
        gx, gy, gw, gh = self.gen_x.value(), self.gen_y.value(), self.gen_w.value(), self.gen_h.value()
        cv2.rectangle(img_with_box, (gx, gy), (gx + gw, gy + gh), (255, 150, 0), 2)
        
        # Specific CNR ROI - Red
        cx, cy, cw, ch = self.cnr_x.value(), self.cnr_y.value(), self.cnr_w.value(), self.cnr_h.value()
        cv2.rectangle(img_with_box, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
        
        # Specific Noise Floor - Green
        nx, ny, nw, nh = self.noise_x.value(), self.noise_y.value(), self.noise_w.value(), self.noise_h.value()
        cv2.rectangle(img_with_box, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

        self.display_image(img_with_box)

    def calculate_general_features(self, roi):
        flat_roi = roi.ravel()
        mean_val = np.mean(flat_roi)
        std_val = np.std(flat_roi)
        
        hist, _ = np.histogram(flat_roi, bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy_val = -np.sum(hist_norm * np.log2(hist_norm))
        
        return mean_val, std_val, entropy_val

    def calculate_cnr(self, cnr_roi, noise_roi):
        mean_cnr = np.mean(cnr_roi)
        mean_noise = np.mean(noise_roi)
        std_noise = np.std(noise_roi)
        
        # Zero division error prevention
        if std_noise == 0:
            std_noise = 1e-6
            
        cnr_val = abs(mean_cnr - mean_noise) / std_noise
        return cnr_val

    def run_analysis(self):
        if not self.file_paths:
            QMessageBox.warning(self, "Warning", "Please load images first!")
            return

        labels = []
        hist_list = []
        features_list = {
            'Mean': [],
            'STD': [],
            'CNR': [],
            'Entropy': []
        }

        # UI에서 좌표값 가져오기
        gx, gy, gw, gh = self.gen_x.value(), self.gen_y.value(), self.gen_w.value(), self.gen_h.value()
        cx, cy, cw, ch = self.cnr_x.value(), self.cnr_y.value(), self.cnr_w.value(), self.cnr_h.value()
        nx, ny, nw, nh = self.noise_x.value(), self.noise_y.value(), self.noise_w.value(), self.noise_h.value()

        for idx, path in enumerate(self.file_paths):
            file_name = os.path.basename(path)
            labels.append(f"Img_{idx+1}")

            img_array = np.fromfile(path, np.uint8)
            cv_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            img_h, img_w = cv_img.shape[:2]
            
            # ROI 크롭 (인덱스 에러 방지)
            gen_roi = cv_img[gy:min(gy+gh, img_h), gx:min(gx+gw, img_w)]
            cnr_roi = cv_img[cy:min(cy+ch, img_h), cx:min(cx+cw, img_w)]
            noise_roi = cv_img[ny:min(ny+nh, img_h), nx:min(nx+nw, img_w)]

            # 1. General 히스토그램 연산
            hist = cv2.calcHist([gen_roi], [0], None, [256], [0, 256]).ravel()
            hist_list.append((file_name, hist))

            # 2. Features 연산 (Mean, STD, Entropy)
            mean_val, std_val, entropy_val = self.calculate_general_features(gen_roi)
            features_list['Mean'].append(mean_val)
            features_list['STD'].append(std_val)
            features_list['Entropy'].append(entropy_val)
            
            # 3. CNR 연산
            cnr_val = self.calculate_cnr(cnr_roi, noise_roi)
            features_list['CNR'].append(cnr_val)

        # --- 그래프 서식 설정 및 그리기 ---
        self.ax_hist.clear()
        self.ax_feat.clear()

        colors = plt.cm.jet(np.linspace(0, 1, len(hist_list)))
        for i, (fname, hist) in enumerate(hist_list):
            self.ax_hist.plot(hist, color=colors[i], alpha=0.7, label=labels[i])
            
        self.ax_hist.set_title("Overlapped Histograms (General ROI)")
        self.ax_hist.set_xlabel("Pixel Intensity")
        self.ax_hist.set_ylabel("Pixel Count")
        self.ax_hist.set_xlim([0, 256])
        if len(hist_list) <= 10:
            self.ax_hist.legend(loc='upper right', fontsize='small')

        # ---------------------------------------------------------
        # (기존) x_indices = range(len(labels)) 부터 아래 영역을 통째로 교체
        # ---------------------------------------------------------
        
        # 흑백 인쇄 시 구분을 위한 지표별 마커 모양 정의
        marker_dict = {
            'Mean': 'o',      # 동그라미
            'STD': '^',       # 세모 (위)
            'CNR': 'v',       # 역세모 (아래)
            'Entropy': '*'    # 별표
        }

        x_indices = range(len(labels))
        for feat_name, values in features_list.items():
            # 딕셔너리에서 지표에 맞는 마커 모양을 가져옴
            marker_style = marker_dict.get(feat_name, 'o') 
            #'o'의 역할: 뒤에 붙은 'o'는 만약 딕셔너리에 정의되지 않은 알 수 없는 지표 이름이 들어왔을 때 
            # 프로그램이 에러(KeyError)를 내며 멈추는 것을 방지하기 위해, "key가 없으면 기본값으로 동그라미('o')를 써라"라고 지정해 둔 예비값(Default)
            
            # marker 파라미터 적용 및 가시성을 위해 markersize 추가
            self.ax_feat.plot(x_indices, values, marker = marker_style, markersize = 8, label = feat_name)

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