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
from scipy.signal import butter, hilbert, filtfilt

# 실험 장비 및 환경 설정 관리 클래스
@dataclass(slots=True)
class ExperimentConfig :
	sampling_rate:float = 1e9 #샘플링 속도(기본: 1 GHz = 1,000,000,000 Hz)
	probe_center_freq:float=45e6 #(기본: 45 MHz)
	bit_depth:int=15 # ADC Data Range (+- 2^15)
	
	#SETP3 : 필터링 기본 설정값(Mhz)
	filter_lowcut_Mhz : float = 20.0 #DC 오프셋 및 아주 낮은 진동 노이즈 제거
	filter_highcut_Mhz : float = 70.0#초고주파 백그라운드 노이즈만 제거
	filter_order : int= 2 #차수(Order)를 4 -> 2로 낮추면 천이 경계가 더 완만해져 원 신호 변형 최소화

	#step4 : Align Mathod
	align_method : str = "envelope_peak" # "pos_max", "neg_min", "cross_corr", "envelope_peak"
	align_pre_samples : int = 100    #align 기준점 이전 샘플 개수
	align_post_samples: int = 500   #align 기준점 이후 샘플 개수
	
	#step5-1 : TGC
	tgc_enable : bool = False
	tgc_start_sample_from_align : int = 0
	tgc_slope_dB : float = 0.02 #샘플당 증폭개인(dB/Sample)

    #Log Compression & Dynamic Range Setting
	log_cmp_alpha : float = 0.003 # Log Scale 계수 (1 + alpha * env)
	log_cmp_dynamic_range_dB : float = 35.0 #Dynamic Range dB
	

	#신규 : ref.csv 파일 경로 추가
	ref_template_csv_path : str = "Document 26.09.01/ref.csv"

	#A Beam의 샘플 갯수
	number_of_samples_in_Whole_Abeam : int = 0

	#step6 : Phase Inverse 검색에서 하드코딩 제외
	phase_inv_search_start_offset_from_align : int = 100
	phase_inv_search_end_offset_from_align : int = 480
	phase_inv_neg_threshold : float = -0.4 #음의 피크 최소 자체 조건
	phase_inv_roi_ratio : float = 1.0 # ROI 내 Max Peak 대비 비율
	phase_inv_whole_pos_ratio : float = 0.15 # 전체 신호 Pos Peak 대비 비율

    

# ==========================================
# 2. 3D Cube Data Engine (2D Align ndarray 기반)
# ==========================================

class UltrasoundCubeEngine : 
	def __init__(self, config: ExperimentConfig) -> None:
		self.config = config

		# 3D CUBE Dimensions
		self.num_rows :int =0
		self.num_cols : int = 0
		self.num_samples : int = 0

		#3D Cubes
		self.raw_cube : Optional[np.ndarray] = None
		self.filtered_cube : Optional[np.ndarray] = None
		self.env_cube : Optional[np.ndarray] = None
		self.roi_cube_8bit : Optional[np.ndarray] = None #B-Scan : 이건 실시간 추출

		# 2D Align Maps (Rows, Cols): envelope_peak, cross_corr 2개만 관리
		self.align_map_envelope_peak : Optional[np.ndarray] = None
		self.align_map_cross_corr : Optional[np.ndarray] = None

		#2D Phase Inverse Map (Rows, Cols)
		self.phase_inv_map : Optional[np.ndarray] = None

		#Shared Axes & Ref Signal
		self.shared_fft_freqs_MHz : Optional[np.ndarray] = None
		self.shared_sample_indices : Optional[np.ndarray] = None
		self.ref_template : Optional[np.ndarray] = None

	def load_files_to_cube(self, file_paths : List[str]) -> bool :

		if not file_paths :
			return False

		try :
			
			#CUBE 차원 정보 설정 및 클래스 멤버 변수 저장
			self.num_rows = len(file_paths)
			df_first = pd.read_csv(file_paths[0], header=None)
			self.num_cols = len (df_first)
			self.num_samples = len(df_first.iloc[0].dropna().values)
			
			#3D Cube 메모리 할당 (np.int16으로 메모리 50% 절감)
			self.raw_cube = np.zeros((self.num_cols,self.num_cols,self.num_samples), dtype = np.int16)
			
			#x축들 생성 : 일반 인덱스 및 주파수 축 생성
			self.shared_sample_indices = np.arange(self.num_samples, dtype=int)
			time_dist = 1.0 / self.config.sampling_rate
			freq_hz = np.fft.rfftfreq(self.num_samples, d = time_dist)
			self.shared_fft_freqs_MHz = freq_hz/ 1e6
			
			#Ref 템플릿 로드
			self._load_ref_template()
			
			#전체 3D 큐브 파이프라인 연산
			self.process_cube_pipleline()
			return True

		except Exception as e:
			print(f"csv파일들->cube 과정에서 문제가 발생 : {e}")
			return False

	def _load_ref_template(self) : 
		path = self.config.ref_template_csv_path
		if os.path.exists(path):
			try:
				df = pd.read_csv(path, header=None, nrows = 1)
				self.ref_template = df.iloc[0].dropna().values.astype(np.float32)
			except Exception as e:
				print(f"Ref template load  과정에서 문제 발생 : {e}")
				self.ref_template = None

	def apply_bandpass_filter(self) -> np.ndarray:
		#Raw CUBE (int16)에 Butter 필터 적용 후, float32로 반환
		if self.raw_cube is None :
			raise ValueError("raw cube 가 로드되지 않았습니다")
		nyquist = 0.5 *self.config.sampling_rate
		low = max(0.001, min((self.config.filter_lowcut_Mhz * 1e6) / nyquist, 0.98))
		high = max(0.002, min((self.config.filter_highcut_Mhz * 1e6) / nyquist, 0.99))

		if low >= high : 
			high = min(low+0.01, 0.99)		

		b,a = butter(self.config.filter_order, [low, high], btype='band')
		_filtered_cube = filtfilt(b, a, self.raw_cube, axis=-1).astype(np.float32)
		return _filtered_cube


	def extract_envelope(self, input_cube : np.ndarray) -> np.ndarray : 
		#필터링된 CUBE 신호로부터, 힐베르트 엔벨롭 추출
		if input_cube is None:
			raise ValueError("엔벨롭 대상 큐브가 없는 오류 입니다")
		return np.abs(hilbert(input_cube,axis=-1)).astype(np.float32)

	def apply_tgc(self, roi_signal_cube : np.ndarray) -> np.ndarray : 

		#ROI Signl CUBE에 Depth TGC 적용
		if not self.config.tgc_enable : 
			return roi_signal_cube

		roi_len = roi_signal_cube.shaple[-1]
		indices = np.arange(roi_len)
		depth_offset = np.maximum(0, indices - self.config.tgc_start_sample_from_align)
		gain_dB = depth_offset * self.config.tgc_slope_dB
		tgc_gain = (10.0 ** (gain_dB / 20.0)).astype(np.float32)

		# In-place 곱셈 연산으로 메모리 효율 유지
		roi_signal_cube *= tgc_gain
		return roi_signal_cube

	def convert_to_8bit_log(self, roi_signal_cube: np.ndarray) -> np.ndarray:
		#"""Config Dynamic Range 파라미터를 적용한 Envelope 및 8-bit Log Compression"""
		roi_env = np.abs(hilbert(roi_signal_cube, axis=-1))
		data_safe = np.maximum(roi_env,0.0)

		alpha = self.config.log_cmp_alpha
		dr_dB = self.config.log_cmp_dynamic_range_dB

		data_log = 20.0 * np.log10(1.0 + alpha * data_safe)

		max_val = np.max(data_log)
		min_cutoff = max_val - dr_dB
		data_log = np.clip(data_log, min_cutoff, max_val)

		norm_data = (data_log - min_cutoff) / dr_dB
		_data_8bit =(norm_data * 255.0).astype(np.uint8)
		return _data_8bit
	
	

	def process_cube_pipeline(self) :
		#3D 큐브 전체 연산 파이프라이
		if self.raw_cube is None :
			raise ValueError("파이프라인 실패. raw cube 가 로드되지 않았습니다")

		#step1 BAND PASS
		self.filtered_cube = self.apply_bandpass_filter()

		#step2 Envelope Extracet
		self.env_cube = self.extract_envelope(self.filtered_cube)

		#step3 2D Aligh Index Map 2개 구성 (envelope peak & cross corr)
		self.compute_align_maps()

		#step4 선택된 Align method 기반 ROI 3D 큐브 추출 & TGC & Log Compression
		self.update_roi_cube()


	def compute_align_maps(self) -> None:
		#1. Envelope Peak 방식 : Envelope의 Max Index  추출
		self.align_map_envelope_peak = np.argmax(self.env_cube, axis=-1).astype(int)

		#2. Cross Correlation 방식 : Ref 필수
		self.align_map_cross_corr = np.zeros((self.num_rows,self.num_cols),dtype=int)
		self.phase_inv_map = np.full((self.num_rows,self.num_cols),-1,dtype=int)

		if self.ref_template is not None:
			inv_start = self.config.phase_inv_search_start_offset_from_align
			inv_end = self.config.phase_inv_search_end_offset_from_align

			for r in range(self.num_rows):
				for c in range(self.num_cols):
					whole_sig = self.filtered_cube[r, c]
					corr = np.correlate(whole_sig, self.ref_template, mode='same')
					pos_idx = int(np.argmax(corr))
					self.align_map_cross_corr[r, c] = pos_idx

					# Phase Inversion 검출 (len(sig) 기준 경계)
					s_idx = pos_idx + inv_start
					e_idx = min(pos_idx + inv_end, len(whole_sig))

					if s_idx < e_idx: #정상 검색 범위임
						roi_corr = corr[s_idx:e_idx]
						min_rel_idx = np.argmin(roi_corr) #argmin 인덱스
						neg_val = roi_corr[min_rel_idx]
						pos_val_in_whole_sig = corr[pos_idx]
						max_val_roi = np.max(roi_corr) #max 값 자체

						if (neg_val < -0.4) and (abs(neg_val) > max_val_roi * 1.0) and (abs(neg_val) > pos_val_in_whole_sig * 0.15):
							self.phase_inv_map[r,c] = s_idx + min_rel_idx
		else:
			self.align_map_cross_corr = self.align_map_envelope_peak.copy()

	def current_align_map(self) -> np.ndarray:
		"""현재 선택된 Align 방식에 따른 2D Align Index Map 반환"""
		if self.config.align_method == 'cross_corr' :
			return self.align_map_cross_corr
		return self.align_map_envelope_peak

	def update_roi_cube(self) :
		pre = self.config.align_pre_samples
		post = self.config.align_post_samples
		roi_len = pre + post

		roi_signal_cube = np.zeros((self.num_rows, self.num_cols, roi_len), dtype=np.float32)
		active_align_map = self.get_current_align_map()

		#2D Align Map 좌표 기준으로 Boundary-safe ROI 슬라이싱
		for r in range(self.num_rows):
			for c in range(self.num_cols):
				a_idx = active_align_map[r, c]

				# CUBE 원본에서의 시작/끝 범위
				r_start = a_idx - pre
				r_end = a_idx + post

				# 실제 CUBE 원본 데이터 경계(Boundary) 제한
				s_start = max(0, r_start)
				s_end = min(self.num_samples, r_end)

				# roi_signal_cube 내부에 복사될 위치 (Offset 계산)
				t_start = s_start - r_start  # 0 이상이며, pre 값을 초과하지 않음
				t_end = t_start + (s_end - s_start)  # 항상 양수이며, roi_len(pre+post) 이하

				# 경계 유효성 검사 후 데이터 대입 (미대입 영역은 자동으로 0.0 Zero-Padding 유지)
				if s_start < s_end:
					roi_signal_cube[r, c, t_start:t_end] = self.filtered_cube[r, c, s_start:s_end]
				else:
					pass# zero padding

		#TGC 적용
		if self.config.tgc_enable:
			indices = np.arange(roi_len)
			depth_offset = np.maximum(0, indices - self.config.tgc_start_sample_from_align)
			gain_dB = depth_offset * self.config.tgc_slope_dB
			tgc_gain = (10.0 ** (gain_dB / 20.0)).astype(np.float32)
			roi_signal_cube *= tgc_gain

		# ROI Envelope & 8-bit Log Compression
		roi_env = np.abs(hilbert(roi_signal_cube, axis=-1))
		data_safe = np.maximum(roi_env, 0.0)
		data_log = 20.0 * np.log10(1.0 + 0.003 * data_safe)
		max_val = np.max(data_log)
		min_cutoff = max_val - 35.0
		data_log = np.clip(data_log, min_cutoff, max_val)
		norm_data = (data_log - min_cutoff) / 35.0
		self.roi_cube_8bit = (norm_data * 255.0).astype(np.uint8)

# ==========================================
# 3. GUI Processor (Tkinter Interface)
# ==========================================