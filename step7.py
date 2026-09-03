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
        self.env_cube = np.abs(hilbert(self.filtered_cube, axis=-1)).astype(np.float32)

        # ---------------------------------------------------------
        # Step 3 & 4. Align & ROI 갱신
        # ---------------------------------------------------------
        self.compute_align_maps()
        self.update_roi_cube()