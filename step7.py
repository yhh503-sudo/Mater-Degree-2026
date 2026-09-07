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
	
	#SETP3 : 필터링 기본 설정값(MHz)
	filter_lowcut_MHz : float = 0.0 #DC 오프셋 및 아주 낮은 진동 노이즈 제거
	filter_highcut_MHz : float = 0.0#초고주파 백그라운드 노이즈만 제거
	filter_order : int= 2 #차수(Order)를 4 -> 2로 낮추면 천이 경계가 더 완만해져 원 신호 변형 최소화

	#step4 : Align Mathod
	align_method : str = "envelope_peak" # "pos_max", "neg_min", "cross_corr", "envelope_peak"
	align_pre_samples : int = 100    #align 기준점 이전 샘플 개수
	align_post_samples: int = 500   #align 기준점 이후 샘플 개수
	
	#step5-1 : TGC
	tgc_enable : bool = True
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


	def __post_init__(self): # 직후에 자동으로 실행되도록 약속된 특수 메서드
		#probe center freq 기반으로 필터 컷오프 주파수 동적 계산
		fc_mhz = self.probe_center_freq / 1e6
		self.filter_lowcut_MHz =max(0.0, fc_mhz / 3.0)
		self.filter_highcut_MHz = fc_mhz * 2.0

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
		self.filtered_cube : Optional[np.ndarray] = None# float32
		self.env_cube : Optional[np.ndarray] = None		# float32
		self.roi_cube_8bit_for_B : Optional[np.ndarray] = None #B-Scan : 이건 실시간 추출
		self.roi_cube_float_for_A : Optional[np.ndarray] = None # ROI Float32 CUBE 추가
		self.roi_env_cube_float_for_A : Optional[np.ndarray] = None 


		# 2D Align Maps (Rows, Cols): envelope_peak, cross_corr 2개만 관리
		self.align_map_envelope_peak : Optional[np.ndarray] = None
		self.align_map_cross_corr : Optional[np.ndarray] = None

		#2D Phase Inverse Map (Rows, Cols)
		self.phase_inv_map : Optional[np.ndarray] = None

		#3D FFT Magnitude Cubes (미리 선언)
		self.raw_fft_mag_cube : Optional[np.ndarray] = None
		self.filtered_fft_mag_cube : Optional[np.ndarray] = None #float32

		#Shared Axes & Ref Signal
		self.shared_fft_freqs_MHz : Optional[np.ndarray] = None
		self.shared_sample_indices : Optional[np.ndarray] = None
		self.ref_template : Optional[np.ndarray] = None
		self.file_paths : List[str] = []
		

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
			self.raw_cube = np.zeros((self.num_rows,self.num_cols,self.num_samples), dtype = np.int16)

			for row_idx, fpath in enumerate(file_paths) : 
				# ✅ 과도한 슬라이싱 방어 코드 및 이중 for문 제거 -> 한꺼번에 Vectorized 2D 할당
				df = pd.read_csv(fpath, header = None).dropna(axis=1, how='all')
				self.raw_cube[row_idx] = df.values.astype(np.int16)


			#x축들 생성 : 일반 인덱스 및 주파수 축 생성
			self.shared_sample_indices = np.arange(self.num_samples, dtype=int)
			time_dist = 1.0 / self.config.sampling_rate
			freq_hz = np.fft.rfftfreq(self.num_samples, d = time_dist)
			self.shared_fft_freqs_MHz = freq_hz/ 1e6
			
			#Ref 템플릿 로드
			self._load_ref_template()
			
			#전체 3D 큐브 파이프라인 연산
			self.process_cube_pipeline()
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
		low = max(0.001, min((self.config.filter_lowcut_MHz * 1e6) / nyquist, 0.98))
		high = max(0.002, min((self.config.filter_highcut_MHz * 1e6) / nyquist, 0.99))

		if low >= high : 
			high = min(low + 0.01, 0.99)		

		b,a = butter(self.config.filter_order, [low, high], btype='band')
		_filtered_cube = filtfilt(b, a, self.raw_cube, axis=-1).astype(np.float32)
		return _filtered_cube


	def extract_envelope(self, input_cube : np.ndarray) -> np.ndarray : 
		#필터링된 CUBE 신호로부터, 힐베르트 엔벨롭 추출
		if input_cube is None:
			raise ValueError("엔벨롭 대상 큐브가 없는 오류 입니다")
		_evn = np.abs(hilbert(input_cube,axis=-1)).astype(np.float32)
		return _evn
	

	def apply_tgc(self, roi_signal_cube : np.ndarray) -> np.ndarray : 

		#ROI Signl CUBE에 Depth TGC 적용
		if not self.config.tgc_enable : 
			return roi_signal_cube

		roi_len = roi_signal_cube.shape[-1]
		indices = np.arange(roi_len)
		depth_offset = np.maximum(0, indices - self.config.tgc_start_sample_from_align)
		gain_dB = depth_offset * self.config.tgc_slope_dB
		tgc_gain = (10.0 ** (gain_dB / 20.0)).astype(np.float32)

		# In-place 곱셈 연산으로 메모리 효율 유지
		roi_signal_cube *= tgc_gain
		return roi_signal_cube

	def compute_fft_cubes(self) -> None:
		#"""Raw 및 Filtered 3D Cube 전체에 대해 FFT Magnitudes를 사전 계산하여 저장"""
		if self.raw_cube is None or self.filtered_cube is None:
			raise ValueError("FFT 계산 실패: Raw 또는 Filtered CUBE가 준비되지 않았습니다.")

		# Numpy Vectorized FFT (마지막 축인 Sample 축에 대해 실수 FFT 수행)
		# np.abs()를 미리 적용.복소수 스펙트럼 대신 크기(Magnitude) 3D 배열로 저장
		self.raw_fft_mag_cube = np.abs(np.fft.rfft(self.raw_cube, axis=-1)/ self.num_samples).astype(np.float32)
		self.filtered_fft_mag_cube = np.abs(np.fft.rfft(self.filtered_cube, axis=-1)/ self.num_samples).astype(np.float32)


	def convert_to_8bit_log(self, roi_signal_cube: np.ndarray) -> np.ndarray:
		#"""Config Dynamic Range 파라미터를 적용한 Envelope 및 8-bit Log Compression"""
		roi_env = np.abs(hilbert(roi_signal_cube, axis = -1))
		data_safe = np.maximum(roi_env, 0.0)

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

		#step3 FFT 3D CUBE 사전 연산 
		self.compute_fft_cubes()


		#step4 2D Aligh Index Map 2개 구성 (envelope peak & cross corr)
		self.compute_align_maps()


		#step5 선택된 Align method 기반 ROI 3D 큐브 추출 & TGC & Log Compression
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

			th_neg = self.config.phase_inv_neg_threshold
			th_roi_ratio = self.config.phase_inv_roi_ratio
			th_pos_ratio = self.config.phase_inv_whole_pos_ratio 

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

						#Config 임계값 기반 판정
						cond1 = neg_val < th_neg
						cond2 = abs(neg_val) > (max_val_roi * th_roi_ratio)
						cond3 = abs(neg_val) > (pos_val_in_whole_sig * th_pos_ratio)

						if cond1 and cond2 and cond3 : 
						#if (neg_val < -0.4) and (abs(neg_val) > max_val_roi * 1.0) and (abs(neg_val) > pos_val_in_whole_sig * 0.15):
							self.phase_inv_map[r,c] = s_idx + min_rel_idx
		else:
			self.align_map_cross_corr = self.align_map_envelope_peak.copy()

	def get_current_align_map(self) -> np.ndarray:
		"""현재 선택된 Align 방식에 따른 2D Align Index Map 반환"""
		if self.config.align_method == 'cross_corr' :
			return self.align_map_cross_corr
		return self.align_map_envelope_peak

	def update_roi_cube(self) :
		pre = self.config.align_pre_samples
		post = self.config.align_post_samples
		roi_len = pre + post

		roi_signal_cube = np.zeros((self.num_rows, self.num_cols, roi_len), dtype = np.float32)
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

		# 2. TGC 적용 후 Float CUBE 저장				
		self.roi_cube_float_for_A = self.apply_tgc(roi_signal_cube)

		self.roi_env_cube_float_for_A = np.abs(hilbert(self.roi_cube_float_for_A))

		# 3. B-Scan용 8-bit Log CUBE 생성
		self.roi_cube_8bit_for_B = self.convert_to_8bit_log(self.roi_cube_float_for_A)

# ==========================================
# 3. GUI Processor (Tkinter Interface)
# ==========================================

class UltrasoundSignalViewer:
	def __init__(self) : 
		self.window = tk.Tk()
		self.window.title("Ultrasound Signal Processor : B Scan Rows Analyzer")
		self.window.geometry("1280x700")

		self.config = ExperimentConfig()
		self.engine = UltrasoundCubeEngine(self.config)

		self.current_row_idx = 0
		self.current_col_idx = 0
		self.view_mode_var = tk.StringVar(value = 'raw')
		self.align_method_var = tk.StringVar(value = self.config.align_method)

		#step6 :  최적화 & Phase Inverse 변수 정의
		self.bscan_img_display : Optional[matplotlib.AxesImage] = None # 화면 패널 레이어 AxesImage 객체
		self.line_bscan_cursor : Optional[Line2D] = None 	# B-Scan 커서
		self.bscan_background = None						# Blitting  기법용 캡처 버퍼
		self.last_ascan_update_time = 0.0					# 쓰트롤링 타임 스탬프

		self.phase_inverse_var =tk.StringVar(value='no_apply') 	#Phase Inverse 토글 기본값
		self.bscan_2d_gray : Optional[np.ndarray] = None 		# 1채널 흑백 버퍼 캐시

		self.create_widgets()

	def create_widgets(self) : 

		#Control Panel Main Frame
		control_frame = ttk.LabelFrame(self.window,text = 'Control Panel', padding =6)
		control_frame.pack(side=tk.TOP, fill = tk.X, padx=10, pady =5)

		#Row 0 Controls

		btn_open = ttk.Button(control_frame, text = 'Open CSVs', command = self.open_csvs)
		btn_open.grid(row=0,column=0,padx=(0,5),pady=2)

		ttk.Label(control_frame, text= 'Row Index ', font=("Segoe UI", 11, "bold")).grid(row=0, column=1, padx=(0, 5), pady=2)
		self.spin_row = ttk.Spinbox(control_frame,from_=0, to=0, width=6, command=self.on_row_change)
		self.spin_row.grid(row=0,column=2,padx=(0,15),pady=2)

		self.lbl_loaded_info = ttk.Label(control_frame,text='Loaded : No files',font=("Segoe UI", 9, "italic"), foreground="green")
		self.lbl_loaded_info.grid(row=0,column=3,padx=(0,20),pady=2,sticky='w')

		#Sperator 1
		ttk.Separator(control_frame,orient='vertical').grid(row=0, column=4, rowspan=2, sticky="ns", padx=10)

		#Align method Frame
		ttk.Label(control_frame,text='Align Method:',font=("Segoe UI", 9, "bold")).grid(row=0, column=5, padx=(0, 5), pady=2)
		ttk.Radiobutton(control_frame,text='Evelope Peak',variable=self.align_method_var,value='envelope_leak',command=self.on_align_change).grid(row=0, column=6, padx=3, pady=2)
		ttk.Radiobutton(control_frame, text ='Cross corr',variable=self.align_method_var,value='cross_corr',command=self.on_align_change).grid(row=0, column=7, padx=3, pady=2)

		#Sperator 2
		ttk.Separator(control_frame, orient="vertical").grid(row=0, column=8, rowspan=2, sticky="ns", padx=10)

		#Phase Inverse Control Group
		ttk.Label(control_frame, text = 'Phase Inverse :', font=("Segoe UI", 9, "bold")).grid(row=0, column=9, padx=(0, 5), pady=2)
		ttk.Radiobutton(control_frame,text="Blue Apply", variable=self.phase_inverse_var, value = "blue_apply", command =self.on_phase_inv_toggle).grid(row=0,column=10,padx=3,pady=2)
		ttk.Radiobutton(control_frame,text="No Apply",variable=self.phase_inverse_var,value="no_apply",command=self.on_phase_inv_toggle).grid(row=0,column=11,padx=3,pady=2,sticky='w')

		#Row 1 Controls
		sub_row1_frame = ttk.Frame(control_frame)
		sub_row1_frame.grid(row=1,column=0,columnspan=4,sticky='w',pady=(5,0))

		ttk.Label(sub_row1_frame, text = 'Col Index').pack(side=tk.LEFT,padx=(5,5))
		# Col 전용 이벤트 함수 분리
		self.spin_col = ttk.Spinbox(sub_row1_frame,from_=0,to=0,width=6,command=self.on_col_change)
		self.spin_col.pack(side=tk.LEFT, padx=(0, 15))
				
		#동적 필터 범위 표기 적용
		low_str = f"{self.config.filter_lowcut_MHz:.1f}"
		high_str = f"{self.config.filter_highcut_MHz:.1f}"
		ttk.Radiobutton(sub_row1_frame,text=f'Filtered {low_str}-{high_str}MHz',variable=self.view_mode_var,value='filtered',command =self.update_ui).pack(side=tk.LEFT,padx=(0,15))
		ttk.Radiobutton(sub_row1_frame,text='Raw Data',variable=self.view_mode_var,value='raw',command=self.update_ui).pack(side=tk.LEFT)

		#Plot Layout
		plot_frame = ttk.Frame(self.window)
		plot_frame.pack(side=tk.TOP, fill=tk.BOTH,expand=True,padx=10,pady=5)

		self.fig = matplotlib.Figure(figsize=(12,6), dpi=100)
		gs = matplotlib.GridSpec(3,2,figure=self.fig,width_ratios=[1,1.2])

		self.ax_roi = self.fig.add_subplot(gs[0,0])
		self.ax_whole = self.fig.add_subplot(gs[1,0])
		self.ax_fft = self.fig.add_subplot(gs[2,0])
		self.ax_bscan = self.fig.add_subplot(gs[:,1])

		self.setup_plots()

		self.canvas = matplotlib.FigureCanvasTkAgg(self.fig, master = plot_frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
		self.canvas.mpl_connect('motion_notify_event',self.on_bscan_hover)

		toolbar = NavigationToolbar2Tk(self.canvas,plot_frame)
		toolbar.update()


	
	def setup_plots(self) : #플롯 세부 설정 위임 : 중간에 초기화들에 사용
	
		#ROX Axis
		self.ax_roi.set_title("ROI Align Signal", fontsize=9, fontweight='bold')
		self.ax_roi.grid(True, linestyle='--', alpha=0.5)
		self.line_roi_sig_for_A = self.ax_roi.plot([], [], color='#1f77b4', lw=1.0, label='Signal')[0]
		self.line_roi_env = self.ax_roi.plot([], [], color='#ff7f0e', lw=1.0, ls='--', label='Envelope')[0]
		self.ax_roi.legend(loc='upper right', fontsize = 7)
		self.ax_roi.set_ylim(-32768,32768)
	
		#Whole AScan AXis
		self.ax_whole.set_title('Raw Ascan', fontsize=9,fontsize=9, fontweight='bold')
		self.ax_whole.grid(True, linestyle='--', alpha=0.5)
		self.line_whole_sig = self.ax_whole.plot([], [], color='#1f77b4', lw=0.8)[0]
		self.line_whole_env = self.ax_whole.plot([], [], color='#ff7f0e', lw=1.0, ls='--')[0]
		self.line_align_mark = self.ax_whole.axvline(x=0, color='red', ls=':', lw=1)
		self.line_inv_mark = self.ax_whole.axvline(x=0, color='purple', ls=':', lw=1, visible=False)
		self.ax_whole.set_ylim(-32768,32768)
	
		#FFT Axis
		self.ax_fft.set_title("Raw FFT : Peak",fontsize=9, fontweight='bold')
		self.ax_fft.grid(True, linestyle='--', alpha=0.5)
		self.line_fft = self.ax_fft.plot([], [], color='#d62728', lw=1.0)[0]
		self.line_fft_peak = self.ax_fft.axvline(x=0, color='green', ls='--', lw=1)
	
		#B Scan Axis
		self.ax_bscan.set_title("B SCAN", fontsize=11, fontweight='bold')
		self.ax_bscan.set_ylabel("Depth Samples")
		self.ax_bscan.set_xlabel("Column")
		self.line_bscan_cursor = self.ax_bscan.axvline(x=0, color='#00ff00', lw=1.5, visible=False)
	
		self.fig.tight_layout()
	
	def open_csvs(self):
		paths = tk.filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
		if not paths : return
	
		if self.engine.load_files_to_cube(list(paths)):
			self.spin_row.config(from_=0, to=self.engine.num_rows - 1)
			self.spin_col.config(from_=0, to=self.engine.num_cols - 1)
	
			self.spin_row.delete(0,tk.END); self.spin_row.insert(0, "0")
			self.spin_col.delete(0, tk.END); self.spin_col.insert(0, "0")
	
			self.update_loaded_label()
	
			#축 범위 한정(최초 1회만 바인딩)
			self.ax_whole.set_xlim(0, self.engine.num_samples)
			self.line_whole_sig.set_xdata(self.engine.shared_sample_indices)
			self.line_whole_env.set_xdata(self.engine.shared_sample_indices)
			self.ax_roi.set_xlim(-self.config.align_pre_samples, self.config.align_post_samples)
	
			#FFT 그래프 : xlimt 동적 계산 (Center F의 1/3~2배)
			self.ax_fft.set_xlim(self.config.filter_lowcut_MHz,self.config.filter_highcut_MHz)
			self.line_fft.set_xdata(self.engine.shared_fft_freqs_MHz)
	
			roi_x  = np.arange(-self.config.align_pre_samples, self.config.align_post_samples)
			self.ax_roi.set_xlim(-self.config.align_pre_samples, self.config.align_post_samples)
			self.line_roi_sig_for_A.set_xdata(roi_x)
	
			self.render_bscan()
			self.update_ui()
	
	def update_loaded_label(self) : 
		if not self.engine.file_paths : 
			return
		filename = os.path.basename(self.engine.file_paths[self.current_row_idx])
		self.lbl_loaded_info.config(text = f"Loaded : {filename}, {self.engine.num_cols} cols", foreground="green")
	
	def render_bscan(self) : 
	
		#1. B Scan 1채널 흑백 버퍼 캐싱 및 Phaser Inverse RGB 오버레이
		if self.engine.roi_cube_8bit_for_B is None : 
			return
		self.bscan_2d_gray = self.engine.roi_cube_8bit_for_B[self.current_row_idx].T #전치
		h, w = self.bscan_2d_gray.shape
	
		pre = self.config.align_pre_samples
		post = self.config.align_post_samples
		extent = [0, w-1, post, -pre]
	
		#2. Phase Inverser 토글 조건 분기 (메모리 재할당 최소화)
	
		cond1 = self.phase_inverse_var.get() == "blue_apply"
		cond2 = self.engine.phase_inv_map is not None
		is_blue_apply = cond1 and cond2
	
		if is_blue_apply :
			# RGB 3채널 배열 생성 (오버레이 적용 시에만 동적 할당)
			display_data = np.stack([self.bscan_2d_gray] * 3, axis = -1)
			active_align_map = self.engine.get_current_align_map()
	
			## Vectorized 또는 범위 검증 루프
			for c in range(w) : 
				inv_abs_idx = self.engine.phase_inv_map[self.current_row_idx,c]
				if inv_abs_idx != -1:
					align_abs_idx = active_align_map[self.current_row_idx, c]
					rel_idx = inv_abs_idx - align_abs_idx + pre #이래야 ROI(-100,500)에 맞음
					if 0 <= rel_idx < h : #제외 : ROI 표시 시작점(align_abs_idx - pre)보다 더 앞쪽에 찍힌 경우
						display_data[rel_idx,c] = [0,0,255] #파란색 마킹
		else:
			#no apply 상태일 떄는 원본 2d grey 버퍼를 그대로 참조 (메모리 복사 없음)
			display_data = self.bscan_2d_gray
	
		#3. AxesImage 갱신 및 캔버스 동기화
	
		# 이미지 객체 재사용 (Image Object Reuse)
		if self.bscan_img_display is not None:
			self.bscan_img_display.set_data(display_data)
			self.bscan_img_display.set_extent(extent)
	
		else: #처음 딱 한번 글일 떄 시행됨
			self.bscan_img_display = self.ax_bscan.imshow(
				display_data, 
				cmap = 'gray' if display_data.ndim == 2 else None,
				aspect = 'auto', origin = 'upper', extent = extent
			)
	
			self.line_bscan_cursor.set_visible(True)
	
			#4. 전체 화면 갱신 후, 초록색 커서 이동용 '배경 비트맵' 최신화
			self.canvas.draw()
			#Blitting 기법용 배경 비트맵 캡쳐
			self.bscan_background = self. canvas.copy_from_bbox(self.ax_bscan.bbox)
	
	def on_row_change(self) : 
		#ROW 변경시 : B-San 전체 재 랜더링 + A-Scan 업데이트
		try:
			self.current_row_idx = int(self.spin_row.get())
			self.update_loaded_label()
			self.render_bscan()
			self.update_ui()
		except ValueError : 
			pass
	
	def on_col_change(self):
		#col 변경시 : 커서 및 a-scan 그래프만 경량 업데이트
		try : 
			self.current_col_idx = int(self.spin_col.get())
			self.update_ui()
		except ValueError:
			pass
	
	def on_align_change(self) : 
		self.config.align_method = self.align_method_var.get()
		self.engine.update_roi_cube()
		self.render_bscan()
		self.update_ui()
	
	def on_phase_inv_toggle(self):
		self.render_bscan()
		self.update_ui()
	
	def update_ui(self) :  #전체 plot들 업데이트
	
		"""경량화된 실시간 업데이트 루틴 (Y축 전용 교체)"""
		if self.engine.raw_cube is None : 
			print(f"UI 업데이트 실패 : raw cube가 없음")
			return
		r,c = self.current_row_idx, self.current_col_idx
		mode = self.view_mode_var.get()
	
		if mode == 'raw' : 
			sig = self.engine.raw_cube[r,c]
			env = self.engine.env_cube[r,c]
			fft_mag = self.engine.raw_fft_mag_cube[r,c] # 사전 계산된 FFT 슬라이싱
		else:
			sig = self.engine.filtered_cube[r,c]
			env = self.engine.env_cube[r,c]
			fft_mag = self.engine.filtered_fft_mag_cube[r,c] # 사전 계산된 FFT 슬라이싱
	
		#1. Whole Ascan plot
		self.line_whole_sig.set_ydata(sig)
		self.line_whole_env.set_ydata(env)
	
		## Blitting으로 초록색 세로 커서 라인 빠른 업데이트
		active_map = self.engine.get_current_align_map()
		align_idx = active_map[r,c]
		self.line_align_mark.set_xdata([align_idx,align_idx])
		inv_idx = self.engine.phase_inv_map[r,c]
	
		if inv_idx != -1:
			self.line_inv_mark.set_xdata([inv_idx,inv_idx])
			self.line_inv_mark.set_visible(True)
			self.ax_roi.set_title(f"ROI Align : {align_idx}, Inverse : {inv_idx}", fontsize=9, fontweight='bold')
			
		else : 
			self.line_inv_mark.set_visible(False)
			self.ax_roi.set_title(f"ROI Align : {align_idx}", fontsize=9, fontweight='bold')
			
		#2. FFT Spectrum : 사전 계산된 배열 슬라이싱 사용
		self.line_fft.set_ydata(fft_mag)
		peak_freq_idx = np.argmax(fft_mag)
		peak_freq_MHz = self.engine.shared_fft_freqs_MHz[peak_freq_idx]
		self.ax_fft.set_title(f"FFT: Peak={peak_freq_MHz:.1f}MHz", fontsize=9, fontweight='bold')
		self.line_fft_peak.set_xdata([peak_freq_MHz, peak_freq_MHz])
	
		max_mag = np.max(fft_mag)
		self.ax_fft.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
	
		#3. ROI Signal(Y축만 교체)		
		roi_sig_data = self.engine.roi_cube_float_for_A[r,c]
		self.line_roi_sig_for_A.set_ydata(roi_sig_data)
		roi_env_data =self.engine.roi_env_cube_float_for_A[r,c]
		self.line_roi_env.set_ydata(roi_env_data)
	
		#4. B Scan Cursor UPdate : Blitting 기법
		self.line_bscan_cursor.set_xdata([c,c])
	
		if self.bscan_background is not None:
			self.canvas.restore_region(self.bscan_background)
			self.ax_bscan.draw_artist(self.line_bscan_cursor)
			self.canvas.blit(self.ax_bscan.bbox)
		else:
			self.canvas.draw_idle()
	
	def on_bscan_hover(self,event) : 
		#ctrl + 마우스 이동 시 쓰트롤링 업데이트
		if event.inaxes == self.ax_bscan and event.key == 'control' :
			if event.xdata is not None :
				col = int(round(event.xdata))
				if 0 <= col < self.engine.num_cols :
					current_time = time.time()
					if current_time - self.last_ascan_update_time > 0.1 :
						self.last_ascan_update_time = current_time
						self.current_col_idx = col
						self.spin_col.delete(0, tk.END)
						self.spin_col.insert(0, str(col))
						self.update_ui()
	def run(self) :
		self.window.mainloop()
	
	
if __name__ == "__main__":
	app = UltrasoundSignalViewer()
	app.run()

