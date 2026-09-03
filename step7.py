import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib. lines import Line2D
import ctypes
import time

#SciPy 신호처리 모듈
from scipy.signal import butter, hilbert, filtfilt, sosfilt
from typing import Optional, NamedTuple, Tuple 

# ==========================================
# 1. Configuration Class
# ==========================================
@dataclass(slots=True)
class ExperimentConfig:
	sampling_rate: float = 1e9  # 1 GHz
	probe_center_freq: float = 45e6  # 45 MHz
    bit_depth: int = 15
    
    # Filter Cutoff (MHz)
	filter_lowcut_Mhz: float = 20.0
	filter_highcut_Mhz: float = 70.0
    filter_order: int = 2

    # Align Settings ('envelope_peak' or 'cross_corr')
    align_method: str = "envelope_peak"
    align_pre_samples: int = 100
    align_post_samples: int = 500
    
    # TGC Settings
    tgc_enable: bool = False
    tgc_start_sample_from_align: int = 0
    tgc_slope_dB: float = 0.02

    # Ref File Path
    ref_template_csv_path: str = "Document 26.09.01/ref.csv"

    # Phase Inversion Search Offsets
    phase_inv_search_start_offset_from_align: int = 100
    phase_inv_search_end_offset_from_align: int = 480

# ==========================================
# 2. 3D Cube Data Engine (2D Align ndarray 기반)
# ==========================================
class UltrasoundCubeEngine:
	def __init__(self, config: ExperimentConfig):
		self.config = config

		#3D Cube Dimension
        self.num_rows : int = 0
        self.num_cols : int = 0
        self.num_samples : int = 0
		
		# 3D Cubes: (Rows, Cols, Samples)
		self.raw_cube: Optional[np.ndarray] = None
		self.filtered_cube: Optional[np.ndarray] = None
		self.env_cube: Optional[np.ndarray] = None
		self.roi_cube_8bit: Optional[np.ndarray] = None
        
        # 2D Align Maps (Rows, Cols): envelope_peak, cross_corr 2개만 관리
		self.align_map_envelope_peak: Optional[np.ndarray] = None
		self.align_map_cross_corr: Optional[np.ndarray] = None
        
        # 2D Phase Inverse Map (Rows, Cols)
		self.phase_inv_map: Optional[np.ndarray] = None
        
        # Shared Axes & Ref Signal
		self.shared_fft_freqs_Mhz: Optional[np.ndarray] = None#self.line_whole_fft.set_xdata
		self.shared_sample_indices: Optional[np.ndarray] = None #self.line_whole_signal.set_xdata
		self.ref_template: Optional[np.ndarray] = None # 1D
        
	def load_files_to_cube(self, file_paths: List[str]) -> bool :
		if not file_paths:return False

		num_files = len(file_paths)
        
        # 첫 번째 파일 기준 규격 확인
		df_first = pd.read_csv(file_paths[0], header=None)
    	num_cols = len(df_first)
        num_samples = len(df_first.iloc[0].dropna().values)

        # 3D CUBE 초기화: Shape = (Rows, Cols, Samples)
        self.raw_cube = np.zeros((num_files, num_cols, num_samples), dtype=np.int16)

        for row_idx, fpath in enumerate(file_paths):
            df = pd.read_csv(fpath, header=None)
            for col_idx, (_, row) in enumerate(df.iterrows()):
                vals = row.dropna().values.astype(np.int16)
                cur_len = min(num_samples, len(vals))
                self.raw_cube[row_idx, col_idx, :cur_len] = vals[:cur_len]

        # X축 인덱스 및 주파수 생성
        self.shared_sample_indices = np.arange(num_samples, dtype=int)
        time_dist = 1.0 / self.config.sampling_rate
        freqs_hz = np.fft.rfftfreq(num_samples, d = time_dist)
        self.shared_fft_freqs_Mhz = freqs_hz / 1e6

        # Ref 템플릿 로드 : 연산용으로 float32 유지
        self._load_ref_template()
        
        # 전체 3D CUBE 파이프라인 연산 수행
        self.process_cube_pipeline()
        return True

    def apply_bandpass_filter(self) -> np.ndarray:
        """Raw CUBE 데이터에 Butterworth Bandpass Filter 적용"""
        if self.raw_cube is None:
            raise ValueError("raw_cube가 로드되지 않았습니다.")

		nyquist = 0.5 * self.config.sampling_rate
        low = max(0.001, min((self.config.filter_lowcut_Mhz * 1e6) / nyquist, 0.98))
        high = max(0.002, min((self.config.filter_highcut_Mhz * 1e6) / nyquist, 0.99))
        if low >= high:
            high = min(low + 0.01, 0.99)

        b, a = butter(self.config.filter_order, [low, high], btype='band')
        
        # int16 raw_cube를 입력받아 float32 반환
        return filtfilt(b, a, self.raw_cube, axis=-1).astype(np.float32)

	def extract_envelope(self, input_cube: np.ndarray) -> np.ndarray:
            """필터링된 CUBE 신호로부터 Hilbert Transform을 사용하여 Envelope 추출"""
        if input_cube is None:
            raise ValueError("입력 CUBE 데이터가 None입니다.")

        return np.abs(hilbert(input_cube, axis=-1)).astype(np.float32)

    def process_cube_pipeline(self):
        if self.raw_cube is None:
            return

        # ---------------------------------------------------------
        # Step 1. Bandpass Filtering
        # np.int16 raw_cube를 연산 시점에 float32/float64로 필터링
        # ---------------------------------------------------------
       
        self.filtered_cube = self.apply_bandpass_filter()

        # ---------------------------------------------------------
        # Step 2. Envelope Extract
        # ---------------------------------------------------------
        self.env_cube = self.extract_envelope(self.filtered_cube)

        # ---------------------------------------------------------
        # Step 3. 2D Align Index Map 생성 
        # ---------------------------------------------------------
        self.compute_align_maps()

		# Step 4. ROI 추출 & TGC & Log Compression
        self.update_roi_cube()

def compute_align_maps(self):

	  #2가지만 사용 : 필터&엔벨롭된 파형에 대해 peak max / ref가지고 corr
    num_rows, num_cols, num_samples = self.raw_cube.shape

	# 1) Envelope Peak 방식:
    # Bandpass -> Hilbert Envelope 거친 3D 배열에서 샘플 축(axis=-1) 기준 Max Index 도출
    self.align_map_envelope_peak = np.argmax(self.env_cube, axis=-1).astype(int)

	# 2) Cross Correlation 방식 (Ref 템플릿 존재 시):
    # 초기화
    self.align_map_cross_corr = np.zeros((num_rows, num_cols), dtype=int)
    self.phase_inv_map = np.full((num_rows, num_cols), -1, dtype=int) #기본값 -1

	if self.ref_template is not None:
        inv_start = self.config.phase_inv_search_start_offset_from_align
        inv_end = self.config.phase_inv_search_end_offset_from_align

		for row_csv in range(num_rows):
            for c in range(num_cols):
                sig = self.filtered_cube[row_csv, c]
                corr = np.correlate(sig, self.ref_template, mode='same')
                pos_idx = int(np.argmax(corr))
                self.align_map_cross_corr[row_csv, c] = pos_idx

				# Phase Inversion 검출 (ROI 검색)
                s_idx = pos_idx + inv_start
                e_idx = min(pos_idx + inv_end, len(sig))

				if s_idx < e_idx:
                    roi_corr = corr[s_idx:e_idx]
					min_rel_idx = np.argmin(roi_corr)
					neg_val = roi_corr[min_rel_idx]
					pos_val = corr[pos_idx]
					max_val_roi = np.max(roi_corr)

					_cond1 = neg_val < -0.4
					_cond2 = abs(neg_val) > max_val_roi * 1.0
					_cond3 = abs(neg_val) > pos_val * 0.15

					if _cond1 and _cond2 and _cond3 : 
						self.phase_inv_map[row_csv,c] = s_idx + min_rel_idx
					self.phase_inv_map[row_csv, c] = s_idx + min_rel_idx

	else:
            # ref 템플릿이 없을 경우 envelope_peak 결과를 대체 복사
            self.align_map_cross_corr = self.align_map_envelope_peak.copy()

def get_current_align_map(self) -> np.ndarray :
       #현재 선택된 align method 방식에 의해서, 2D int MAP 반환
       if self.config.align_method == 'cross_corr' :
              return self.align_map_cross_corr
       #기본값 envelop_peak
       return self.align_map_envelope_peak

def update_roi_cube(self):
       pre = self.config.align_pre_samples
       post = self.config.align_post_samples
       roi_len = pre + post

	   roi_signal_cube = np.zeros((self.num_rows, self.num_cols, roi_len), dtype=np.float32)
       active_align_map = self.get_current_align_map()

	   #2D Align Map 기준으로 Fancy Indexing 처리하여 ROI 슬라이싱
	for r in range(self.num_rows):
       for c in range(self.num_cols):
		   a_idx = active_align_map[r,c]
                 
           # CUBE 원본에서의 시작/끝 범위
           r_start = a_idx - pre
           r_end = a_idx + post

            # 실제 CUBE 원본에서, 데이터 경계(Boundary) 제한
	   		s_start = max(0,r_start)
       		s_end = min(self.total_samples, r_end)

            #roi_signal 내부에 복사될 위치 : Offset을 미리 계산
            t_start = s_start - r_start #무조건 0 이상이지만, pre 보다 커질 수도 없다.
            t_end = t_start + (s_end - s_start) #양수이고, pre+pre+post 보다 늘 작다.

            #경계성 유효성 검사 후, 데이터 대입
            if s_start < s_end :
              roi_signal[r,c,t_start:t_end]  = = self.filtered_cube[r, c, s_start:s_end]