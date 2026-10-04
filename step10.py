import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict
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


# ==========================================
# 0. 인프라 Layer (Publisher)
# ==========================================
class DataEventPublisher : 

	#데이터 및 Config 변경 이벤트를 구독자에게 알리는 이벤트 발행자 클래스
	#엔진은 UI가 누구인지, 존재하는지조차 모릅니다. 메모리는 똑같이 쓰지만 연락망(Publisher)만 이용합니다.
	#UI를 Tkinter에서 Web으로 바꾸든, CLI로 바꾸든, UI를 3개 더 추가하든 Engine 코드는 단 한 줄도 수정할 필요가 없습니다.

	def __init__(self):
		self._subscribers : Dict[str, List[Callable[[],None]]] = {} ## 단순히 { "이벤트이름": [콜백함수1, 콜백함수2] } 형태
		#Callable = _CallableType(collections.abc.Callable, 2)

	def subscribe(self, event_type_in : str, callback_in : Callable[[], None]) -> None : 
		#특정 이벤트에 대한 구독자(Callback) 등록
		if event_type_in not in self._subscribers :
			self._subscribers[event_type_in] = []
		self._subscribers[event_type_in].append(callback_in)

	def notify(self, _event_type_in : str) -> None : 
		'''등록된 구독자들에게 이벤트 발생 통보'''
		if _event_type_in in self._subscribers : 
			for _callback in self._subscribers[_event_type_in] :
					_callback() # 리스트 0번부터 순차적으로 실행



# ==========================================
# 1. Model Layer (Data & State)
# ==========================================

@dataclass
class AppState:
	#1. UI 활성 선택 상태
	active_row : int = 0
	active_col : int = 0
	active_align_map : Optional[np.ndarray] = field(default=None, init=False)

	#2. 하드웨어 수집 환경
	sampling_rate : float = 1e9 # 샘플링 속도 (1 GHz)
	probe_center_freq : float = 45e6 #탐촉자 중심주파수 (45MHz)
	bit_depth : int = 15 #ADC Data Range (+- 2^15)

	#3. 신호 처리 및 정렬 설정
	filter_lowcut_MHz : float = 0.0
	filter_highcut_MHz : float = 0.0
	filter_order : int = 2
	align_method : str = 'envelope_peak'
	align_pre_samples : int = 100
	align_post_samples : int = 500

	#4. TGC & Log Compression 설정
	tgc_enable : bool = True
	tgc_start_sample_from_align : int = 0
	tgc_slope_dB : float = 0.02
	log_cmp_alpha : float = 0.003
	log_cmp_dynamic_range_dB : float = 35.0

	#5. 외부 문서 및 반전 탐지 설정
	ref_template_csv_path : str = "Document 26.09.01/ref.csv"
	number_of_samples_in_whold_Abeam : int = 0
	phase_inv_search_start_offset_from_align : int = 0
	phase_inv_search_end_offset_from_align : int = 0
	phase_inv_neg_threshold : float = -0.4
	phase_inv_roi_ratio : float = 1.0
	phase_inv_whole_pos_ratio : float = 0.15

	#6. C-Scan 설정
	cscan_gate_start : int = 1
	cscan_gate_end : int  = 180
	cscan_stretch_mode : str = 'absolute'
	cscan_merge_mode : str = 'max'

	def __post_init__(self) : 
		self.update_filter_cutoffs()
	
	def update_filter_cutoffs(self) -> None:
		fc_MHz = self.probe_center_freq/ 1e6
		self.filter_lowcut_MHz = max(0, fc_MHz/3.0)
		self.filter_highcut_MHz = fc_MHz * 2.0

@dataclass
class CLayer : 
	data_8bit_2d_map : np.ndarray
	depth_start : int
	depth_end : int
	gate_mode : str = 'max'
	slice_index : int = 0 #여기에 대해선 의문

# [개선] 순수 데이터 컨테이너 (slots=True 메모리 최적화 유지)
@dataclass(slots=True) 
class UltrasoundCubeData:
	
	file_paths : List[str] = field(default_factory=list)

	num_rows : int = 0
	num_cols : int = 0
	num_samples : int = 0

	# 3D Cubes (볼륨 데이터)
	raw_3d_cube :Optional[np.ndarray] = None #int16
	filtered_3d_cube : Optional[np.ndarray] = None #float32
	env_ed_cube : Optional[np.ndarray] = None #float32

	# ROI : B-Scan, C-Scan 재료  3D Cubes
	roi_3d_cube_float_for_A : Optional[np.ndarray] = None #float32
	roi_3d_env_cube_float_for_A : Optional[np.ndarray] = None #flot32
	roi_3d_cube_8bit_for_B : Optional[np.ndarray] = None #uint8

	#3D FFT Mangitude Cubes
	raw_fft_mag_3d_cube : Optional[np.ndarray] = None #float32
	filtered_fft_mag_3d_cube : Optional[np.ndarray] = None #float32

	#2D Maps (계산 결과 참조)
	align_map_envelope_peak : Optional[np.ndarray] = None #int16
	align_map_cross_corr : Optional[np.ndarray] = None #int16
	phase_inv_map : Optional[np.ndarray] = None # int16이면 충분함

	#Shared Axes & Ref Singal
	ref_template : Optional[np.ndarray] = None
	shared_sample_indices : Optional[np.ndarray] = None
	shared_fft_freqs_MHz : Optional[np.ndarray] = None

	#C-Scan 합성
	c_layers : List[CLayer] = field(default_factory=list)
	active_cscan_map : Optional[np.ndarray] = None
	
	
#3. 신호처리 및 연산 전담 엔진
# ==========================================
# 2. Control Layer (Engine)
# ==========================================

class UltrasoundProcessorEngine:
	def __init__(self, data :UltrasoundCubeData, publisher_in : DataEventPublisher) -> None :

		self.data : UltrasoundCubeData = data
		#변경 : Engine이 가변 상태(State)를 소유하고 통제
		self.state : AppState = AppState()
		self.publisher : DataEventPublisher = publisher_in #주입

	def load_files_to_cube(self, file_paths : List[str])->bool : 
		if not file_paths:
			return False

		try :		
			#CUBE 차원 정보 설정 및 클래스 멤버 변수 저장
			self.data.file_paths = file_paths
			self.data.num_rows = len(file_paths)
			df_first = pd.read_csv(file_paths[0], header=None)
			self.data.num_cols = len (df_first)
			self.data.num_samples = len(df_first.iloc[0].dropna().values)
			
			#3D Cube 메모리 할당 (np.int16으로 메모리 50% 절감)
			self.data.raw_3d_cube = np.zeros((self.data.num_rows,self.data.num_cols,self.data.num_samples), dtype = np.int16)

			for row_idx, fpath in enumerate(file_paths) : 
				# ✅ 과도한 슬라이싱 방어 코드 및 이중 for문 제거 -> 한꺼번에 Vectorized 2D 할당
				df = pd.read_csv(fpath, header = None).dropna(axis=1, how='all')
				self.data.raw_3d_cube[row_idx] = df.values.astype(np.int16)


			#x축들 생성 : 일반 인덱스 및 주파수 축 생성
			self.data.shared_sample_indices = np.arange(self.data.num_samples, dtype=int)
			time_dist = 1.0 / self.state.sampling_rate
			freq_hz = np.fft.rfftfreq(self.data.num_samples, d = time_dist)
			self.data.shared_fft_freqs_MHz = freq_hz/ 1e6
			
			#Ref 템플릿 로드
			self._load_ref_template()
			
			#전체 3D 큐브 파이프라인 연산
			self.process_cube_pipeline()

			#변경] 데이터 로드 후, 기본 상태로 초기화
			self.state.active_row = 0
			self.state.active_col = 0

			#Event Notify] 전체 데이터로 로드 완료
			self.publisher.notify("DATA_LOADED")

			return True

		except Exception as e:
			print(f"csv파일들-> cube 과정에서 문제가 발생 : {e}")
			return False


	def _load_ref_template(self) -> None :
		_path = self.state.ref_template_csv_path
		if os.path.exists(_path):
			try:
				df = pd.read_csv(_path, header=None, nrows=1)
				self.data.ref_template = df.iloc[0].dropna().values.astype(np.float32)
			except Exception as e:
				print(f"Ref template 로드 오류: {e}")
				self.data.ref_template = None


	def process_cube_pipeline(self) -> None :
		if self.data.raw_3d_cube is None:
			raise ValueError("Raw cube가 할당되지 않았습니다.")

		#1. 밴드패스 필터
		self.data.filtered_3d_cube = self.apply_bandpass_filter(self.data.raw_3d_cube)
		
		#2. 필터 거친 후 -> 엔벨롭
		self.data.env_3d_cube = self.extract_envelope(self.data.filtered_3d_cube)

		#3. self.compute_fft_cube()
		self.data.raw_fft_mag_3d_cube = self.compute_fft_cube(self.data.raw_3d_cube, data_samples = self.data.num_samples)
		self.data.filtered_fft_mag_3d_cube = self.compute_fft_cube(self.data.filtered_3d_cube, data_samples = self.data.num_samples)
		
		#4. Align 인덱스들 계산 & Inverse 계산
		self.compute_align_maps()

		#5. roi cube : a, b, c 스캔 만듦
		self.update_roi_cube_A_B_C()


	def set_active_selection(self, row_in : int, col_in : int) -> None : 

		#사용자 위치 선택 변경 처리 : 핵심 Notify
		_row_changed = (self.state.active_row != row_in)
		_col_changed = (self.state.active_col != col_in)

		self.state.active_row = max(0, min(row_in, self. data.num_rows -1))
		self.state.active_col = max(0, min(col_in, self. data.num_cols -1))

		#Event Notify 위치 변경 발생 통보
		
		if _row_changed :
			self.publisher.notify("ROW_CHANGED") #ROW_CHANGED가 먼저 터져서 B-Scan 이미지를 새로 그린 뒤, SELECTION_CHANGED가 터져서 A-Scan과 커서를 맞춰 그려줍니다.

		if _row_changed or _col_changed :
			self.publisher.notify("SELECTION_CHANGED") #A-Scan Changed가 나을수도 
		#특정 좌표 $(Row, Col)$ 위치의 단일 A-Scan 신호(1D 파형), Spectrum(FFT), 그리고 B-Scan 영상 위의 초록색 커서 선 위치를 이동


	def set_cscan_parameters(self, d_start : int, d_end : int, stretch_mode : str, merge_mode : str) -> None:
		self.state.cscan_gate_start = d_start
		self.state.cscan_gate_end = d_end
		self.state.cscan_stretch_mode = stretch_mode
		self.state.cscan_merge_mode = merge_mode

		if self.data.roi_3d_cube_8bit_for_B is not None:
			self.generate_single_cscan(d_start, d_end)


	def generate_single_cscan(self, d_start : int, d_end : int) -> None : 
		layer = self.extract_cscan_layer(d_start, d_end, gate_mode = self.state.cscan_merge_mode)
		_stretched_map = self.apply_histogram_stretch(layer.data_8bit_2d_map, mode = self.state.cscan_stretch_mode)
		self.data.active_cscan_map = _stretched_map
		self.publisher.notify("CSCAN_UPDATED")
		#Q. CSCAN UPDATED 노티파이 후에, CSCAN PARAMTERS UPDATED 노티파이가 맞는 건가?  중복되는 게 아닌가?
		pass #단일 C-Scan 생성 로직 확장 가능 구간


	def get_bscan_2d_map(self, row_idx: int) -> np.ndarray:
		if self.data.roi_3d_cube_8bit_for_B is None:
			raise ValueError("B-Scan Data Cube 8bit짜리가 준비되지 않았습니다.")
		r = max(0, min(row_idx, self.data.num_rows - 1))
		return self.data.roi_3d_cube_8bit_for_B[r].T

	def extract_cscan_layer(self, depth_start : int, depth_end : int, gate_mode : str = 'max') -> CLayer :

		if self.data.roi_3d_cube_8bit_for_B is None :
			raise ValueError("B-Scan Data Cube(8비트)가 준비되지 않았습니다")

		s_idx = max(0, depth_start + self.state.align_pre_samples)
		e_idx = min(self.data.roi_3d_cube_8bit_for_B.shape[-1], depth_end + self.state.align_pre_samples)

		#8 bit로만 C-SCAN은 연산함. 거의 차이 없음.
		gate_data = self.data.roi_3d_cube_8bit_for_B[:, :, s_idx : e_idx] #해당 Z들만 뽑았네.

		if gate_data.shape[-1] == 0:
			#무슨 예외처리네?
			cscan_2d = np.zeros((self.data.num_rows, self.data.num_cols), dtype=np.uint8)
		
		elif gate_mode == "max" :
			cscan_2d = np.max(gate_data, axis = -1).astype(np.uint8)
		elif gate_mode == "mean" :
			cscan_2d = np.mean(gate_data, axis = -1).astype(np.uint8)
		else : 
			cscan_2d =  np.mean(gate_data, axis = -1).astype(np.uint8)

		return CLayer(data_8bit_2d_map = cscan_2d, depth_start = depth_start, depth_end = depth_end, gate_mode = gate_mode)
	

	def apply_histogram_stretch(self, 
							 cscan_2d : np.ndarray, 
							 mode : str = 'absolute', 
							 abs_range : Tuple[int,int] = (10, 245), 
							 n_std : float = 2.0) -> np.ndarray :

		#연산 위해 : 컨버트
		img_float = cscan_2d.astype(np.float32)

		if mode == 'absolute' :
			min_val, max_val = abs_range

		elif mode == 'relative_std' :
			mean_v = np.mean(img_float)
			std_v = np.std(img_float)
			min_val = max(0.0, mean_v - n_std * std_v)
			max_val = min(255.0, mean_v + n_std * std_v)
			#이에 따르면, n_std가 커질수록, 더 넓은 data Range가 0-255로 맵핑됨
		else :
			min_val, max_val = abs_range

		if max_val <= min_val :  #예외 처리
			max_val = min_val + 1.0

		_std_diff = max_val - min_val
		stretched = ((img_float - min_val) / (_std_diff)) * 255.0
		#이러면, Relative에서 음수도 가능한데? 이걸 clip으로 커버침?
		return np.clip(stretched, 0, 255).astype(np.uint8)

	def set_align_method(self, _method_in : str) -> None :
		#Align 방식 변경 처리
		self.state.align_method = _method_in
		self.update_roi_cube_A_B_C()
		#[Event Notify] ROI 데이터 재계산 완료 통보
		self.publisher.notify("ROI_UPDATED")


	def update_active_align_map_pointer(self) -> None :
		# Active map  포인터 갱신
		if self.state.align_method == 'cross_corr' and self.data.align_map_cross_corr is not None:
			self.state.active_align_map = self.data.align_map_cross_corr
		else :
			self.state.active_align_map = self.data.align_map_envelope_peak


	def apply_bandpass_filter(self, signal : np.ndarray, axis_in : int = -1) -> np.ndarray :
		nyquist : float = 0.5 * self.state.sampling_rate
		low: float = max(0.001, min((self.state.filter_lowcut_MHz * 1e6) / nyquist, 0.98))
		high: float = max(0.002, min((self.state.filter_highcut_MHz * 1e6) / nyquist, 0.99))

		if low >= high:
			high = min(low + 0.01, 0.99)

		b, a = butter(self.state.filter_order, [low, high], btype='band')
		return filtfilt(b, a, signal, axis = axis_in).astype(np.float32)

	
	def extract_envelope(self, signal_array : np.ndarray, axis_in : int = -1) -> np.ndarray :
			return np.abs(hilbert(signal_array, axis=axis_in)).astype(np.float32)

	
	def compute_fft_cube(self, signal_array : np.ndarray, data_samples : int, axis_in : int = -1) -> np.ndarray :
		return np.abs(np.fft.rfft(signal_array, axis = axis_in) / data_samples).astype(np.float32)


	def compute_align_maps(self) -> None:
		#1. Envelope Peak 방식 : Envelope의 Max Index  추출
		self.data.align_map_envelope_peak = np.argmax(self.data.env_3d_cube, axis=-1).astype(int)
		#2. Cross Correlation 방식 : Ref 필수
		self.data.align_map_cross_corr = np.zeros((self.data.num_rows, self.data.num_cols), dtype=int)
		self.data.phase_inv_map = np.full((self.data.num_rows, self.data.num_cols),-1, dtype=int)

		if self.data.ref_template is not None:
			inv_start = self.state.phase_inv_search_start_offset_from_align
			inv_end = self.state.phase_inv_search_end_offset_from_align

			th_neg = self.state.phase_inv_neg_threshold
			th_roi_ratio = self.state.phase_inv_roi_ratio
			th_pos_ratio = self.state.phase_inv_whole_pos_ratio 

			for r in range(self.data.num_rows):
				for c in range(self.data.num_cols):
					whole_sig = self.data.filtered_3d_cube[r, c]
					corr = np.correlate(whole_sig, self.data.ref_template, mode='same')
					pos_idx = int(np.argmax(corr))
					self.data.align_map_cross_corr[r, c] = pos_idx

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
							self.data.phase_inv_map[r,c] = s_idx + min_rel_idx
		else:
			self.data.align_map_cross_corr = self.data.align_map_envelope_peak.copy()


	def apply_tgc(self, roi_signal_3d_cube : np.ndarray) -> np.ndarray : 

		#ROI Signl CUBE에 Depth TGC 적용
		if not self.state.tgc_enable : 
			return roi_signal_3d_cube

		roi_len = roi_signal_3d_cube.shape[-1]
		indices = np.arange(roi_len)
		depth_offset = np.maximum(0, indices - self.state.tgc_start_sample_from_align)
		gain_dB = depth_offset * self.state.tgc_slope_dB
		tgc_gain = (10.0 ** (gain_dB / 20.0)).astype(np.float32)

		# 연산시마다, 증폭을 막기 위해서 변경
		# roi_signal_cube *= tgc_gain
		return roi_signal_3d_cube * tgc_gain


	def convert_to_8bit_log(self, roi_signal_3d_cube: np.ndarray) -> np.ndarray:

		#Config Dynamic Range 파라미터를 적용한 8-bit Log Compression"""
		data_safe = np.maximum(roi_signal_3d_cube, 0.0)
		alpha = self.state.log_cmp_alpha
		dr_dB = self.state.log_cmp_dynamic_range_dB

		data_log = 20.0 * np.log10(1.0 + alpha * data_safe)

		max_val = np.max(data_log)
		min_cutoff = max_val - dr_dB
		data_log = np.clip(data_log, min_cutoff, max_val)
 
		norm_data = (data_log - min_cutoff) / dr_dB
		_data_8bit =(norm_data * 255.0).astype(np.uint8)
		return _data_8bit

	
	def update_roi_cube_A_B_C (self) -> None :

		"""A-Scan, B-Scan, C-Scan 데이터 파이프라인 통합 재구성 메서드"""
		pre : int = self.state.align_pre_samples
		post : int = self.state.align_post_samples
		roi_len : int = pre + post

		roi_3d_signal_32bit_cube : np.ndarray = np.zeros((self.data.num_rows, self.data.num_cols, roi_len), dtype = np.float32)
		self.update_active_align_map_pointer()
		active_align_map : Optional[np.ndarray] = self.state.active_align_map

		#2D Align Map 좌표 기준으로 Boundary-safe ROI 슬라이싱
		for r in range(self.data.num_rows):
			for c in range(self.data.num_cols):

				#전체 다 옮기기

				a_idx = active_align_map[r, c]

				# CUBE 원본에서의 시작/끝 범위
				r_start = a_idx - pre
				r_end = a_idx + post

				# 실제 CUBE 원본 데이터 경계(Boundary) 제한
				s_start = max(0, r_start)
				s_end = min(self.data.num_samples, r_end)

				# roi_3d_signal_cube 내부에 복사될 위치 (Offset 계산)
				t_start = s_start - r_start  # 0 이상이며, pre 값을 초과하지 않음
				t_end = t_start + (s_end - s_start)  # 항상 양수이며, roi_len(pre+post) 이하

				# 경계 유효성 검사 후 데이터 대입 (미대입 영역은 자동으로 0.0 Zero-Padding 유지)				
				if s_start < s_end:
					roi_3d_signal_32bit_cube[r, c, t_start:t_end] = self.data.filtered_3d_cube[r, c, s_start:s_end]

				else:
					pass# zero padding

		# 1. TGC 적용 후 Float 3D CUBE 저장				
		self.data.roi_3d_cube_float_for_A = self.apply_tgc(roi_3d_signal_32bit_cube) #self.roi_cube_float_for_A는 TGC가 기본 적용임


		# 2. 범용 extract_envelope 함수를 통해 ROI Envelope 3D 계산
		self.data.roi_3d_env_cube_float_for_A = self.extract_envelope(self.data.roi_3d_cube_float_for_A)


		# 3. B-Scan용 8-bit Log 3D CUBE 생성
		self.data.roi_3d_cube_8bit_for_B = self.convert_to_8bit_log(self.data.roi_3d_env_cube_float_for_A)


		# 4. Align 재정렬에 맞춰 C-Scan 맵도 자동 재계산
		self.generate_single_cscan(self.state.cscan_gate_start, self.state.cscan_gate_end)


# ==========================================
# 3. GUI Processor (Tkinter Interface) : Subscriber / Event-Driven View
# ==========================================

class UltrasoundSignalViewer:

	#스스로 그래프를 그리는 것이 아니라, Engine에서 알림(Notify)이 날아오면 그 반응으로 화면을 업데이트
	def __init__(self) : 
		self.window = tk.Tk()
		self.window.title("Ultrasound Signal Viewer (Step 10)")
		self.window.geometry("1280x800")

		#인프라 객체 생성 :  Publisher를 전체 시스템이 공유
		self.publisher : DataEventPublisher = DataEventPublisher()
		self.data : UltrasoundCubeData = UltrasoundCubeData()#config = self.config)
		self.engine :UltrasoundProcessorEngine = UltrasoundProcessorEngine(data = self.data, publisher_in=self.publisher)

		self.view_mode_var = tk.StringVar(value = 'raw')
		self.align_method_var = tk.StringVar(value = self.config.align_method)


		#step6 :  최적화 & Phase Inverse 변수 정의
		self.bscan_img_display : Optional[matplotlib.AxesImage] = None # 화면 패널 레이어 AxesImage 객체
		self.line_bscan_cursor : Optional[Line2D] = None 	# B-Scan 커서
		self.last_ascan_update_time = 0.0					# 쓰트롤링 타임 스탬프 : 현재 이것만 사용

		self.phase_inverse_var =tk.StringVar(value='no_apply') 	#Phase Inverse 토글 기본값
		self.bscan_2d_gray : Optional[np.ndarray] = None 		# 1채널 흑백 버퍼 캐시
		self._bscan_rgb_cache : Optional[np.ndarray] = None # UI 렌더링 전용 픽셀 버퍼 캐시 (Model 데이터가 아닌 View 전용)

		self.cscan_merge_var = tk.StringVar(value=self.config.cscan_merge_mode)
		self.cscan_stretch_var = tk.StringVar(value=self.config.cscan_stretch_mode)

		self.cscan_img_display: Optional[matplotlib.image.AxesImage] = None
		self.line_cscan_horiz: Optional[Line2D] = None
		self.line_cscan_vert: Optional[Line2D] = None
		self.last_ascan_update_time = 0.0

		self.create_widgets()
		#구독 패턴 등록
		self.register_event_subscriptions()



	@property
	def state(self) -> AppState :
		return self.engine.state

	# --------------------------------------------------------------------------
    # UI 생성 루틴
    # --------------------------------------------------------------------------


	def create_widgets(self) : 

		#Q. 왜 컴포넌트들이 Col이 1개씩 밀렸죠? 제가 수정해서, 옳게 바꿨습니다.

		#Control Panel Main Frame
		control_frame = ttk.LabelFrame(self.window,text = 'Control Panel', padding =6)
		control_frame.pack(side=tk.TOP, fill = tk.X, padx=10, pady =5)

		# 1. Open Button (2행 높이 통합)
		btn_open = ttk.Button(control_frame, text = 'Open CSVs', command = self.open_csvs)
		btn_open.grid(row=0,column=0,padx=(0,5),pady=2)

		# 2. Status Labels (위: 전체 크기 / 아래: 현재 선택 파일명)

		self.lbl_shape_info = ttk.Label(control_frame,text='0 Rows * 0 Cols', font=("Arial", 12, "bold"), foreground="#2e7d32")
		self.lbl_shape_info.grid(row=0, column=1, padx=10, pady=2, sticky="w")

		self.lbl_loaded_info = ttk.Label(control_frame, text="No files loaded", font=("Arial", 10), foreground="#4caf50")
		self.lbl_loaded_info.grid(row=1, column=1, padx=10, pady=2, sticky="w")

		#Sperator 1
		ttk.Separator(control_frame, orient='vertical').grid(row=0, column=2, rowspan=2, sticky="ns", padx=10)

		#3. Spinboxes
		ttk.Label(control_frame, text= 'Row Index ', font=("Segoe UI", 11, "bold")).grid(row=0, column=3, padx=2, pady=2)
		self.spin_row = ttk.Spinbox(control_frame, from_=0, to=0, width=5, command = self.on_row_change)
		self.spin_row.grid(row=0, column=4, padx=5, pady=2)
		self.spin_row.bind('<Return>', lambda e: self.on_row_change())

		ttk.Label(control_frame,text='Col Index').grid(row=1,column=3,padx=2, pady=2, sticky="e")
		self.spin_col = ttk.Spinbox(control_frame,from_=0,to=0,width=5,command=self.on_col_change)
		self.spin_col.grid(row=1, column=4, padx=5, pady=2)
		self.spin_col.bind('<Return>', lambda e : self.on_col_change())

		#Sperator 2
		ttk.Separator(control_frame, orient='vertical').grid(row=0,column=5,rowspan=2, sticky="ns", padx=10)

		#4. BandPas Radio
		ttk.Label(control_frame, text = 'BandPass', font=("Arial", 10, "bold")).grid(row=0, column=6, rowspan=2, padx=5)
		#동적 필터 범위 표기 적용
		low_str = f"{self.config.filter_lowcut_MHz:.1f}"
		high_str = f"{self.config.filter_highcut_MHz:.1f}"
		ttk.Radiobutton(control_frame, text=f'Filtered {low_str}-{high_str}MHz',variable=self.view_mode_var,value='filtered',command =self.update_ascan_plots).grid(row=0, column=7, padx=5, pady=2, sticky="w")
		ttk.Radiobutton(control_frame, text='Raw Data',variable=self.view_mode_var,value='raw',command=self.update_ascan_plots).grid(row=1, column=7, padx=5, pady=2, sticky="w")

		#Sperator 3
		ttk.Separator(control_frame,orient='vertical').grid(row=0, column= 8, rowspan=2, sticky="ns", padx=10)
		

		#Align method Frame
		ttk.Label(control_frame,text='Align Method:',font=("Segoe UI", 9, "bold")).grid(row=0, column=9, padx=(0, 5), pady=2)
		ttk.Radiobutton(control_frame, text='Evelope Peak',variable = self.align_method_var, value='envelope_peak',command=self.on_align_change).grid(row=0, column=10, padx=3, pady=2, sticky='w')
		ttk.Radiobutton(control_frame, text ='Cross corr',variable = self.align_method_var, value='cross_corr',command=self.on_align_change).grid(row=1, column=10, padx=3, pady=2, sticky='w')
		#라디오 버튼을 클릭하는 순간 "cross_corr"라는 문자열이 self.align_method_var에 즉시 저장 / 그 다음. command=self.on_align_change: 버튼을 클릭해 값이 바뀌었을 때 실행할 콜백 함수

		#Sperator 4
		ttk.Separator(control_frame, orient="vertical").grid(row=0, column = 11, rowspan=2, sticky="ns", padx=10)

		#Phase Inverse Control Group
		ttk.Label(control_frame, text = 'Phase Inverse :', font=("Segoe UI", 9, "bold")).grid(row=0, column=12, padx=(0, 5), pady=2)
		ttk.Radiobutton(control_frame,text="Blue Apply", variable=self.phase_inverse_var, value = "blue_apply", command =self.on_phase_inv_toggle).grid(row=0,column=13,padx=3,pady=2,sticky='w')
		ttk.Radiobutton(control_frame,text="No Apply",variable=self.phase_inverse_var,value="no_apply",command=self.on_phase_inv_toggle).grid(row=1,column=13,padx=3,pady=2,sticky='w')

		#Sperator 5
		ttk.Separator(control_frame, orient="vertical").grid(row=0, column=14, rowspan=2, sticky="ns", padx=8)

		# 5. C-Scan Depth Settings
		ttk.Label(control_frame, text='Depth start', font=("Segoe UI", 9, "bold")).grid(row=0, column=15, padx=2, pady=2, sticky="e")
		self.spin_depth_start = ttk.Spinbox(control_frame, from_=-100, to=500, width=5, command=self.on_cscan_setting_change)
		self.spin_depth_start.grid(row=0, column=16, padx=3, pady=2)
		self.spin_depth_start.delete(0, tk.END); self.spin_depth_start.insert(0, str(self.config.cscan_gate_start))
		self.spin_depth_start.bind('<Return>', lambda e: self.on_cscan_setting_change())

		ttk.Label(control_frame, text='Depth end', font=("Segoe UI", 9, "bold")).grid(row=1, column=15, padx=2, pady=2, sticky="e")
		self.spin_depth_end = ttk.Spinbox(control_frame, from_=-100, to=500, width=5, command=self.on_cscan_setting_change)
		self.spin_depth_end.grid(row=1, column=16, padx=3, pady=2)
		self.spin_depth_end.delete(0, tk.END); self.spin_depth_end.insert(0, str(self.config.cscan_gate_end))
		self.spin_depth_end.bind('<Return>', lambda e: self.on_cscan_setting_change())

		ttk.Separator(control_frame, orient="vertical").grid(row=0, column=17, rowspan=2, sticky="ns", padx=8)

		# 6. Merge Options
		ttk.Label(control_frame, text='Merge', font=("Segoe UI", 9, "bold")).grid(row=0, column=18, rowspan=2, padx=2)
		ttk.Radiobutton(control_frame, text='Max', variable=self.cscan_merge_var, value='max', command=self.on_cscan_setting_change).grid(row=0, column=19, padx=2, pady=2, sticky='w')
		ttk.Radiobutton(control_frame, text='Mean', variable=self.cscan_merge_var, value='mean', command=self.on_cscan_setting_change).grid(row=1, column=19, padx=2, pady=2, sticky='w')

		ttk.Separator(control_frame, orient="vertical").grid(row=0, column=20, rowspan=2, sticky="ns", padx=8)

		# 7. Stretch Options
		ttk.Label(control_frame, text='Stretch', font=("Segoe UI", 9, "bold")).grid(row=0, column=21, rowspan=2, padx=2)
		ttk.Radiobutton(control_frame, text='Abs', variable=self.cscan_stretch_var, value='absolute', command=self.on_cscan_setting_change).grid(row=0, column=22, padx=2, pady=2, sticky='w')
		ttk.Radiobutton(control_frame, text='Relative', variable=self.cscan_stretch_var, value='relative_std', command=self.on_cscan_setting_change).grid(row=1, column=22, padx=2, pady=2, sticky='w')

				
		#Main Plot Layout
		plot_frame = ttk.Frame(self.window)
		plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

		self.fig = matplotlib.figure.Figure(figsize=(13, 7), dpi=100)
		_gs = GridSpec(3,2,figure=self.fig,width_ratios=[1,1.2])

		self.ax_roi = self.fig.add_subplot(_gs[0,0])
		self.ax_whole = self.fig.add_subplot(_gs[1,0])
		self.ax_fft = self.fig.add_subplot(_gs[2,0])
		self.ax_bscan = self.fig.add_subplot(_gs[0:1, 1]) 
		#이게 위치가 맞나? 대략 c-scan  이 2배 더 row 크니. 맞는 듯
		self.ax_cscan = self.fig.add_subplot(_gs[1:3, 1])

		self.setup_plots()

		self.canvas = FigureCanvasTkAgg(self.fig, master = plot_frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)

		# 인터랙션 연동
		self.canvas.mpl_connect('motion_notify_event',self.on_bscan_hover)
		self.canvas.mpl_connect('button_press_event', self.on_cscan_click) 
		#Q.이게 왜 on_cscan으로 연결되지?  b-scan이 아니고?
		#B-SCAN 호버 처럼, C-SCAN 호버를 하자.
		
		toolbar = NavigationToolbar2Tk(self.canvas,plot_frame)
		toolbar.update()
	
	def setup_plots(self) : #플롯 세부 설정 위임 : 중간에 초기화들에 사용
	
		#ROX Axis : 여기에 총 구성요소 2개
		self.ax_roi.set_title("ROI Align Signal", fontsize=9, fontweight='bold')
		self.ax_roi.grid(True, linestyle='--', alpha=0.5)
		self.line_roi_sig_for_A = self.ax_roi.plot([], [], color='#1f77b4', lw=1.0, label='Signal')[0]
		self.line_roi_env = self.ax_roi.plot([], [], color='#ff7f0e', lw=1.0, ls='--', label='Envelope')[0]
		self.ax_roi.legend(loc='upper right', fontsize = 7)
		self.ax_roi.set_ylim(-32768,32768)
	
		#Whole AScan AXis : 여기에 총 구성요소 4개
		self.ax_whole.set_title('Raw Ascan', fontsize=9, fontweight='bold')
		self.ax_whole.grid(True, linestyle='--', alpha=0.5)
		self.line_whole_sig = self.ax_whole.plot([], [], color='#1f77b4', lw=0.8)[0]
		self.line_whole_env = self.ax_whole.plot([], [], color='#ff7f0e', lw=1.0, ls='--')[0]
		self.line_align_mark = self.ax_whole.axvline(x=0, color='red', ls=':', lw=1)
		self.line_inv_mark = self.ax_whole.axvline(x=0, color='purple', ls=':', lw=1, visible=False)
		self.ax_whole.set_ylim(-32768,32768)
	
		#FFT Axis : 여기에 총 구성요소 2개
		self.ax_fft.set_title("Raw FFT : Peak",fontsize=9, fontweight='bold')
		self.ax_fft.grid(True, linestyle='--', alpha=0.5)
		self.line_fft = self.ax_fft.plot([], [], color='#d62728', lw=1.0)[0]
		self.line_fft_peak = self.ax_fft.axvline(x=0, color='green', ls='--', lw=1)
	
		#B Scan Axis : bscan 에는 이미지에, 커서 1개만 존재함
		self.ax_bscan.set_title("B SCAN", fontsize=11, fontweight='bold')
		self.ax_bscan.set_ylabel("Depth Samples")
		self.line_bscan_cursor = self.ax_bscan.axvline(x=0, color='#00ff00', lw=1.5, visible=False)

		#C Scan Axis : 커서가 row, col 총 2개 존재함
		self.ax_cscan.set_title("C-SCAN", fontsize=11, fontweight='bold')
		self.line_cscan_horiz = self.ax_cscan.axhline(y=0, color='#00ff00', lw=1.0, visible=False)
		self.line_cscan_vert = self.ax_cscan.axvline(x=0, color='#00ff00', lw=1.0, visible=False)

		self.fig.tight_layout()


	def register_event_subscriptions(self) :

		# 구독(Subscribe) 등록: 특정 이벤트 시 내 UI 업데이트 메서드들을 연결!
        # ----------------------------------------------------------------------
		# 이벤트 발행 시, 실행될 UI 콜백(Observer) 메서드 등록
		self.publisher.subscribe("DATA_LOADED",self.on_event_datas_loaded)

		# 2. Row 변경 시 실행 (B-Scan 갱신 + 라벨 갱신)
		self.publisher.subscribe("ROW_CHANGED", self.update_loaded_label)
		self.publisher.subscribe("ROW_CHANGED", self.render_bscan)

		self.publisher.subscribe("SELECTION_CHANGED", self.update_ascan_plots)
		#Q. SELECTION_CHANGED 보다, "SELECTED_A-BEAM_CHANGED"가 더 맞는 표현 아닌가요?
		self.publisher.subscribe("ROI_UPDATED", self.on_event_roi_updated)

		self.publisher.subscribe("CSCAN_UPDATED", self.render_cscan)
		self.publisher.subscribe("CSCAN_SETTINGS_CHANGED", self.on_event_cscan_settings_changed)


	# Event Driven Subscriber Callbacks (이벤트 반응 함수들)	
	def on_event_datas_loaded(self) :
		
		#'DATA_LOADED' 이벤트 수신 시 수행 : open_csvs(self) 의 일부를 대체
		self.spin_row.config(from_=0, to = self.data.num_rows -1)
		self.spin_col.config(from_=0, to = self.data.num_cols - 1)

		self.spin_row.delete(0, tk.END); self.spin_row.insert(0, "0")
		self.spin_col.delete(0, tk.END); self.spin_col.insert(0, "0")

		self.lbl_shape_info.config(text=f'{self.data.num_rows}Rows*{self.data.num_cols}Cols')
		self.update_loaded_label()

		self.ax_whole.set_xlim(0, self.data.num_samples)
		self.line_whole_sig.set_xdata(self.data.shared_sample_indices)
		self.line_whole_env.set_xdata(self.data.shared_sample_indices)

		self.line_fft.set_xdata(self.data.shared_fft_freqs_MHz)#데이터 자체는 0Hz부터 나이퀴스트 주파수(Sampling Rate의 절반, 예: 500MHz)까지
		self.ax_fft.set_xlim(self.config.filter_lowcut_MHz, self.config.filter_highcut_MHz)#set_xlim()에 의해 Matplotlib의 화면 출력 범위(시야)만 밴드패스 필터 구간으로 잘라서 보여
				
		_roi_x = np.arange(-self.config.align_pre_samples, self.config.align_post_samples)
		self.ax_roi.set_xlim(-self.config.align_pre_samples, self.config.align_post_samples)
		self.line_roi_sig_for_A.set_xdata(_roi_x)
		self.line_roi_env.set_xdata(_roi_x)

		self.render_bscan()
		self.render_cscan()
		self.update_ascan_plots()

	def on_event_roi_updated(self):

		#'ROI_UPDATED' 이벤트 수신시 수행
		self.render_bscan()
		self.render_cscan()
		self.update_ascan_plots() 
		#Q. 이게 뒤에 존재해도 괜찮은가? Canvas DraW idle이 포함되어 있으니, 마지막에 실행되어야 할수 있음

	def on_event_cscan_settings_changed(self) -> None:
		self.render_cscan()

	#UI Handlers
	def on_cscan_setting_change(self) -> None :
		#사용자가 화면의 Spinbox나 Checkbox 값을 변경했을 때 UI(View)가 직접 호출하는 핸들러입니다.
		try:
			d_start = int(self.spin_depth_start.get())
			d_end = int(self.spin_depth_end.get())
		except ValueError:
			print(f"C-Scan Depth 설정값들이 spinbox들에 잘못 입력되었습니다")
			d_start = self.config.cscan_gate_start
			d_end = self.config.cscan_gate_end

		_merge_mode = self.cscan_merge_var.get()
		_stretch_mode = self.cscan_stretch_var.get()

		self.engine.set_cscan_parameters(d_start, d_end, _merge_mode, _stretch_mode)
		self.publisher.notify("CSCAN_SETTINGS_CHANGED")

#
# UI 이벤트 핸들러 : 사용자 입력 -> Engine 으로 전달하는 통로
#

	def on_event_cscan_settings_changed(self) -> None :
		#Publisher가 "CSCAN_SETTINGS_CHANGED" 이벤트를 방송(Notify)했을 때 실제 렌더링/화면 갱신을 수행하는 콜백(Callback) 함수입니다.
		#반면, 지금처럼 분리해 두면 이벤트 발행(notify)은 단 1줄만 남고, 
		# 각 UI 컴포넌트들이 각자 on_event_cscan_settings_changed() 같은 수신 콜백을 등록하기만
		# 하면 되므로 코드가 매우 깨끗해집니다
		self.render_cscan()


	def on_row_change(self) -> None : 
		# UI 이벤트 핸들러 (사용자 입력 -> Engine으로 전달하는 통로)
		# ROW 변경시 : B-San 전체 재 랜더링 + A-Scan 업데이트
		"""UI에서 스핀박스로 Row 조작 시"""
		try:
			_r = int(self.spin_row.get())
			_c = int(self.spin_col.get())
			# UI가 직접 랜더링하는 것이 아니라 Engine의 위치를 바꾸면,
			# Engine이 알고리즘 판단 후 ROW_CHANGED / SELECTION_CHANGED 이벤트를 던집니다.
			self.engine.set_active_selection(_r,_c)
			#self.update_loaded_label()
		except : 
			print("스핀박스에 row, col 값이 제대로 입력되었는지 확인 필요. 기존 data사용")
			_r = self.state.active_row
			_c = self.state.active_col
			self.engine.set_active_selection(_r,_c)
			pass
	
	def on_col_change(self) -> None :
		# col 변경시 : 커서 및 a-scan 그래프만 경량 업데이트
		# UI에서 스핀박스로 Col 조작시
		try :
			r = int(self.spin_row.get())
			c = int(self.spin_col.get())
			self.engine.set_active_selection(r,c) 
		except :
			print("스핀박스에 row, col 값이 제대로 입력되었는지 확인 필요")
			r = self.data.active_row
			c = self.data.active_col
			self.engine.set_active_selection(r,c) 
			pass

	def on_align_change(self) : 
		#Align 방식 변경시
		self.engine.set_align_method(self.align_method_var.get())

	
	def on_phase_inv_toggle(self) :
		"""Blue 오버레이 토글 시"""
		#self.bscan_img_display : Optional[matplotlib.AxesImage] 바꾸는 것에 지나지 않음
		#UI 클래스 내부 처리이기 떄문에 손 안 댐:구독 패턴(Publisher-Subscriber)은 시스템 전체가 공유해야 하는 '핵심 데이터 상태'가 바뀔 때 사용하는 것이 가장 좋습니다.
		#"시스템의 데이터 상태(Model/Engine State)가 바뀌는 핵심 이벤트만 Publisher를 타고, 화면 표현을 위한 단순 렌더링 옵션(View State)은 View 내부에서 가볍게 처리한다
		self.render_bscan()
		self.update_ascan_plots()
	

	def open_csvs(self) -> None:
		paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
		if not paths :
			return
		else:
			#파일을 읽고 파이프라인 연산이 끝나면, Engine 내부에서 "DATA_LOADED"가 터짐
			self.engine.load_files_to_cube(list(paths))

#
# 화면 랜더링 함수들 (UI 업데이트)
#


	def render_bscan(self) : 

		"""B-Scan 2D 이미지 갱신"""
		#1. B Scan 1채널 흑백 버퍼 캐싱 및 Phaser Inverse RGB 오버레이

		if self.data.roi_cube_8bit_for_B is None :
			raise ValueError("B스캔 버퍼 만들기 실패. 큐브 생성 파이프라인을 실행한 적이 없습니다") 
			return

		r = self.data.active_row
		self.bscan_2d_gray = self.data.roi_cube_8bit_for_B[r].T #전치
		h, w = self.bscan_2d_gray.shape

		#1. 뷰어 규격 변경 시에만 메모리 재할당
		if (self._bscan_rgb_cache is None) or (self._bscan_rgb_cache.shape != (h,w,3)) :
			self._bscan_rgb_cache = np.empty((h,w,3),dtype = np.uint8)

		#2. 캐시 버퍼에 흑백 채널 복사
		self._bscan_rgb_cache [:,:,0] = self.bscan_2d_gray
		self._bscan_rgb_cache [:,:,1] = self.bscan_2d_gray
		self._bscan_rgb_cache [:,:,2] = self.bscan_2d_gray

	
		pre = self.config.align_pre_samples
		post = self.config.align_post_samples
		extent = [0, w-1, post, -pre]


		#2. Phase Inverser 토글 조건 분기 (메모리 재할당 최소화)
		cond1 = self.phase_inverse_var.get() == "blue_apply"
		cond2 = self.data.phase_inv_map is not None
		is_blue_apply = cond1 and cond2

		display_data : Optional[np.ndarray] = None
	
		if is_blue_apply :
			# RGB 3채널 배열 생성 (오버레이 적용 시에만 동적 할당)
			self.engine.update_active_align_map_pointer()
			active_align_map = self.data.active_align_map
	
			## Vectorized 또는 범위 검증 루프
			for c in range(w) : 
				inv_abs_idx = self.data.phase_inv_map[r, c]
				if inv_abs_idx != -1:
					align_abs_idx = active_align_map[r, c]
					rel_idx = inv_abs_idx - align_abs_idx + pre #이래야 ROI(-100,500)에 맞음
					if 0 <= rel_idx < h : #제외 : ROI 표시 시작점(align_abs_idx - pre)보다 더 앞쪽에 찍힌 경우
						self._bscan_rgb_cache[rel_idx,c] = [0,0,255] #파란색 마킹
		else:
			#no apply 상태일 떄는 원본 2d grey 버퍼를 그대로 참조 (메모리 복사 이미함)
			pass

		#3. AxesImage 갱신 및 캔버스 동기화	
		# 이미지 객체 재사용 (Image Object Reuse)
		if self.bscan_img_display is not None:
			self.bscan_img_display.set_data(self._bscan_rgb_cache) #data 교체 끼우기
			self.bscan_img_display.set_extent(extent)
	
		else: #처음 딱 한번 글일 떄 시행됨
			self.bscan_img_display = self.ax_bscan.imshow( #객체 생성
				self._bscan_rgb_cache, 
				aspect = 'auto', 
				origin = 'upper', 
				extent = extent
			)
			self.line_bscan_cursor.set_visible(True)

	def render_cscan(self) :
		if self.data.active_cscan_map is None :
			print(f"C-Scan map 완성된 것 없음")
			return

		if self.cscan_img_display is not None :
			self.cscan_img_display.set_data (self.data.active_cscan_map)
		else :
			# 없으면 생성, 처음에만 실행됨
			self.cscan_img_display = self.ax_cscan.imshow(
				self.data.active_cscan_map,
				cmap='gray',
				aspect='auto',
				origin='upper'
				)
			self.line_cscan_horiz.set_visible(True)
			self.line_cscan_vert.set_visible(True)
		self.canvas.draw_idle()

	def update_ascan_plots(self) -> None :  #전체 plot들 업데이트
	
		#"""경량화된 실시간 업데이트 루틴 (Y축 전용 교체) + 전체 Draw"""
		# SELECTION_CHANGED 발생 시 실행 : A-Scan들 및 커서 위치만 경량 업데이트

		if self.data.raw_cube is None : 
			print(f"UI 업데이트 실패 : raw cube가 없음")
			return
		
		r, c = self.data.active_row, self.data.active_col
		mode = self.view_mode_var.get()

		#"메모리의 r 값과 화면 Spinbox의 글자가 다르면 화면 글자도 r로 바꿔라"라는 UI 동기화 작업
		if str(r) != self.spin_row.get():
			self.spin_row.delete(0, tk.END)
			self.spin_row.insert(0, str(r))
		if str(c) != self.spin_col.get():
			self.spin_col.delete(0, tk.END)
			self.spin_col.insert(0, str(c))

	
		if mode == 'raw' : 
			sig = self.data.raw_3d_cube[r,c]
			env = self.data.env_3d_cube[r,c]
			fft_mag = self.data.raw_fft_mag_3d_cube[r,c] # 사전 계산된 FFT 슬라이싱
		else:
			sig = self.data.filtered_3d_cube[r,c]
			env = self.data.env_3d_cube[r,c]
			fft_mag = self.data.filtered_fft_mag_3d_cube[r,c] # 사전 계산된 FFT 슬라이싱
	
		#1. Whole Ascan plot
		self.line_whole_sig.set_ydata(sig)
		self.line_whole_env.set_ydata(env)
		## 표시 라인 빠른 업데이트 : Align 및 Phase Inverse 마커 위치 업데이트
		self.engine.update_active_align_map_pointer()
		
		align_idx : int = self.data.active_align_map[r, c]
		self.line_align_mark.set_xdata([align_idx, align_idx])
		inv_idx = self.data.phase_inv_map[r,c]
	
		if inv_idx != -1:
			self.line_inv_mark.set_xdata([inv_idx, inv_idx])
			self.line_inv_mark.set_visible(True)
			self.ax_roi.set_title(f"ROI Align : {align_idx}, Inverse : {inv_idx}", fontsize=9, fontweight='bold')
			
		else : 
			self.line_inv_mark.set_visible(False)
			self.ax_roi.set_title(f"ROI Align : {align_idx}", fontsize=9, fontweight='bold')

			
		#2. FFT Spectrum : 사전 계산된 배열 슬라이싱 사용
		self.line_fft.set_ydata(fft_mag)
		peak_freq_idx = np.argmax(fft_mag)
		peak_freq_MHz = self.data.shared_fft_freqs_MHz[peak_freq_idx]
		self.ax_fft.set_title(f"FFT: Peak={peak_freq_MHz:.1f}MHz", fontsize=9, fontweight='bold')
		self.line_fft_peak.set_xdata([peak_freq_MHz, peak_freq_MHz])
	
		max_mag = np.max(fft_mag)
		self.ax_fft.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
	
		#3. ROI Signal(Y축만 교체)
		if self.data.roi_3d_cube_float_for_A is not None : 	
			self.line_roi_sig_for_A.set_ydata(self.data.roi_3d_cube_float_for_A[r,c])

		if self.data.roi_3d_env_cube_float_for_B is not None:
			self.line_roi_env.set_ydata(self.data.roi_3d_env_cube_float_for_B[r, c])
		
		#B-Scan Cursor 위치 갱신 : c에 의해, 실시간 바뀜
		self.line_bscan_cursor.set_xdata([c, c])
		self.line_cscan_horiz.set_ydata([r, r])
		self.line_cscan_vert.set_xdata([c, c])


		self.canvas.draw_idle()#CPU가 Idle(한가한) 상태가 되거나, 이벤트 루프가 돌아올 때 그려집니다.
								#여러 번 호출되어도 마지막 1번만 그려집니다
			

	
	def update_loaded_label(self) : 
		if not self.data.file_paths or self.data.active_row >= len(self.data.file_paths) : 
			print(f"아직 읽은 csv파일들이 없거나, 설정한 값이 가능 row 범위를 넘어갔습니다.")
			return
		_filename = os.path.basename(self.data.file_paths[self.data.active_row])
		self.lbl_loaded_info.config(text = f"{_filename} Loaded", foreground="green")

	def on_mouse_hover(self, event_in) : 
		if event_in.key != 'control' :
			return

		current_time = time.time()
		if current_time - self.last_ascan_update_time < 0.05 :
			return
		if event_in.inaxes == self.ax_bscan and event_in.xdata is not None :
			_col = int(round(event_in.xdata))
			if 0 <= _col < self.data.num_cols :
				self.last_ascan_update_time = current_time
				self.engine.set_active_selection(self.data.active_row, _col)

			elif event_in.inaxes == self.ax_cscan and event_in.xdata is not None and event_in.ydata is not None :
				_col = int(round(event_in.xdata))
				_row = int(round(event_in.ydata))

				if 0<= _row < self.data.num_rows and 0 <= _col < self.data.num_cols :
					self.last_ascan_update_time = current_time
					self.engine.set_active_selection(_row,_col)

	def on_cscan_click(self, event_in) :
		#Q. 그런데, Ctrl 버튼을 누른체 하는 조건이 포함된 게 맞나요? 아니면, 사용자 의도로 보기 힘든데도, 엄청 연산들이 진행될지도요.
		if event_in.inaxes == self.ax_cscan and event_in.xdata is not None and event_in.ydata is not None:
			_col = int(round(event_in.xdata))
			_row = int(round(event_in.ydata))
			self.engine.set_active_selection(_row, _col)

		#Q. B-SCAN은 호버 함수가 있는데, 사실 내가 사용자이면, C-SCAN에서 호버를 더 많이 할 듯요. C-SCAN용 호버도 필요합니다. 거기에도, 시간 0.1초 이상 그런 스트롤링 조건이 필수일듯요

						
	def on_bscan_hover(self, event) : 
		#ctrl + 마우스 이동 시 쓰트롤링 업데이트
		if event.inaxes == self.ax_bscan and event.key == 'control' :
			if event.xdata is not None :
				_col = int(round(event.xdata))
				if 0 <= _col < self.data.num_cols :
					current_time = time.time()
					
					if current_time - self.last_ascan_update_time > 0.1 :

						self.last_ascan_update_time = current_time

						# UI 스핀박스 텍스트 신속 갱신
						self.spin_col.delete(0, tk.END)
						self.spin_col.insert(0, str(_col)) #스핀박스도 바뀜. 그러나, on_col_change()이벤트는 미발생
						# Engine 연산 호출 (이벤트 체인 발동)
						self.engine.set_active_selection(self.data.active_row,_col)

	def run(self) :
		self.window.mainloop()
	
	
if __name__ == "__main__":
	app = UltrasoundSignalViewer()
	app.run()