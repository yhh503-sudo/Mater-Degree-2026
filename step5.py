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
	align_ROI_samples : int = 600 #ROI 샘플 갯수

	#step5-1 : TGC
	tgc_enable : bool = False
	tgc_start_sample_from_align : int = 0
	tgc_slope_dB : float = 0.04 #샘플당 증폭개인(dB/Sample)

	#신규 : ref.csv 파일 경로 추가
	ref_template_csv_path : str = "Material csv(26.07.28)/ref.csv"

	#A Beam의 샘플 갯수
	number_of_samples_in_Whole_Abeam : int = 0

class AlignIndices: # C-Struct형 Align 인덱스 저장 클래스
	__slots__ = ('envelope_peak', 'pos_max', 'neg_min', 'cross_corr') #변수이름의 키
	#__slots__는 파이썬에게 "이 클래스는 유연함(동적 추가)을 포기할 테니, 지정한 변수만 고정해서 C언어의 구조체(struct)처럼 딱딱하게 만들어줘
	def __init__(self):
		self.envelope_peak:Optional[int]=None
		self.pos_max:Optional[int]=None
		self.neg_min:Optional[int]=None
		self.cross_corr:Optional[int]=None

	def get(self, method:str) -> Optional[int]:
		'''Method문자열 이름으로 빠르게 속성값 반환'''
		if method == 'envelope_peak':
			return self.envelope_peak
		elif method == 'pos_max':
			return self.pos_max
		elif method == 'neg_min':
			return self.neg_min
		elif method == 'cross_corr':
			return self.cross_corr
		return None
	

#AScan:단일 AScan 데이터 클래스
class AScan:
	def __init__(self, raw_data_in:np.ndarray, row_index_in:int=0, col_index_in:int=0 ):
		self.raw_data :np.ndarray = raw_data_in
		self.row_index : int = row_index_in# 단일 CSV 시 0으로 고정
		self.col_index : int = col_index_in# CSV 파일 내 행(Row) 번호
		
		# 1. Raw 파형 분석 결과
		self.fft_magnitude:Optional[np.ndarray]=None #set_ydata(self.current_ascan.fft_magnitude)
		self.center_freq_Mhz : float = 0.0
		self.raw_envelope_data:Optional[np.ndarray] = None
		
		#2. Filtered 파형 분석 결과
		self.filtered_data : Optional[np.ndarray] = None
		self.filtered_fft_magnitude : Optional[np.ndarray] = None
		self.filtered_center_freq_Mhz : float = 0.0
		self.filtered_envelope_data : Optional[np.ndarray] = None

		#3. STEP4 : Align 및 600 샘플 ROI 결과 저장 변수
		self.align_indices : AlignIndices = AlignIndices()
		self.ROI_Max : Optional[np.ndarray] = None
		self.ROI_min : Optional[np.ndarray] = None
		self.ROI_Corr : Optional[np.ndarray] = None
		self.ROI_Envelope_Raw : Optional[np.ndarray] = None

		#4. STEP4-2 : Phase Inversion
		self.phase_inverse : Optional[int] = None

		#5. Step5-1 : TGC
		self.tgc_ROI_Max : Optional[np.ndarray] = None
		self.tgc_ROI_min : Optional[np.ndarray] = None
		self.tgc_ROI_Corr : Optional[np.ndarray] = None
		self.tgc_ROI_Envelope : Optional[np.ndarray] = None

		#5. Step5-3 : Envelope
		self.env_tgc_ROI_Max : Optional[np.ndarray] = None
		self.env_tgc_ROI_min : Optional[np.ndarray] = None
		self.env_tgc_ROI_Corr : Optional[np.ndarray] = None
		self.env_tgc_ROI_Envelope : Optional[np.ndarray] = None

		#5. Step5-4 : B-Scan 매핑용 8bit uint 메모리 효율화 변수 추가
		self.roi_bscan_bytes : Optional[np.ndarray] = None # (ROI_samples,) 크기의 uint8 배열


	def get_align_index(self, method : str) -> Optional[int] : 
		'''지정한 어라인 방식의 인덱스 반환'''
		return self.align_indices.get(method)
	 
	# ==========================================
    # 🎯 독립된 개별 신호 처리 메서드 모음
    # ==========================================	

	def apply_tgc(self, start_sample : int = 0, slope_dB : float = 0.005):
		#단순 선형 깊이 감쇄 보정 함수

		if self.filtered_data is None:
			return

		n_samples = len(self.filtered_data)
		indices = np.arange(n_samples)

		#start_sample 이후부터 거리 비례 dB 증폭 적용
		depth_offset = np.maxmimum(0, indices - start_sample)
		gain_dB = depth_offset * slope_dB
		gain_linear = 10.0 ** (gain_dB/20.0) #dB스케일을 선형 스케일로 변환

		#TGC 반영된 ndarray 데이터 생성 및 Envelope 계산
		self.tgc_data = self.filtered_data * gain_linear
		self.tgc_envelope_data = self.extract_envelope(self.tgc_data)


	def apply_tgc_all(self, tgc_gain_ndarray : np.ndarray):
		self.tgc_ROI_Max = self.ROI_Max * tgc_gain_ndarray
		self.tgc_ROI_min = self.ROI_min * tgc_gain_ndarray
		self.tgc_ROI_Corr= self.ROI_Corr * tgc_gain_ndarray
		self.tgc_ROI_Envelope = self.ROI_Envelope_Raw * tgc_gain_ndarray
		pass

	def compute_all_align_indices(self, ref_template : Optional[np.ndarray] = None):
		'''4가지 방식의 Align Index를 __slots__객체 속성에 빠르게 셋팅'''
		if self.filtered_data is None or self.filtered_envelope_data is None:#align 대상은 : 반드시 밴드패스 거친
			return	
		target_signal = self.filtered_data
		target_envelope = self.filtered_envelope_data
		#1. Envelope Peak
		self.align_indices.envelope_peak = int(np.argmax(target_envelope))
		#np.argmax(target_env)"가장 큰 피크가 있는 샘플의 위치 번호"를 반환int : "NumPy 전용 정수(np.int64)를 순수 파이썬 정수(int)로 바꿔서 __slots__에 깔끔하고 안전하게 저장하기 위함"
		#2. Positive Max
		self.align_indices.pos_max = int(np.argmax(target_signal))
		#3. Negative Min
		self.align_indices.neg_min = int(np.argmin(target_signal))
		#4. Cross Correlation (Pattern Matching		
		if ref_template is not None :
			_corr = np.correlate(target_signal, ref_template, mode = 'same')
			pos_idx = int(np.argmax(_corr))
			pos_val = _corr[pos_idx]
			self.align_indices.cross_corr = pos_idx
			#메인 파형 영향권을 벗어난 100~+500 구간 정의
			search_start = pos_idx + 100
			search_end = min(pos_idx + 500, len(_corr))

			if search_start < search_end:	
				#ROI  구간
				_roi_corr = _corr[search_start:search_end]
				_roi_neg_realtive_idx = int(np.argmin(_roi_corr))
				_roi_neg_idx = search_start + _roi_neg_realtive_idx
				_abs_roi_neg_val = abs(_corr[_roi_neg_idx])

				roi_pos_relative_idx = int(np.argmax(_roi_corr))
				roi_pos_idx = search_start + roi_pos_relative_idx
				roi_pos_val = _corr[roi_pos_idx]

				_cond1 = _abs_roi_neg_val > roi_pos_val * 1
				_cond2 = _abs_roi_neg_val > pos_val * 0.15

				#반사파 구간에서 음의 피크가 양의 피크보다 우세할 경우, Phase Inverse
				if _cond1  and _cond2 :
					self.phase_inverse = _roi_neg_idx

		else: #템플릿이 없으면, 기본적으로 envelope_peak 결과를 대입
			self.align_indices.cross_corr = self.align_indices.envelope_peak
		pass

	def compute_fft(self, signal: np.ndarray, fft_freqs_Mhz_in : Optional[np.ndarray]=None) -> Tuple[Optional[np.ndarray], Optional[float]]:
		"""통일 FFT 함수 (Magnitude 배열, Center Frequency 튜플 반환)"""

		n_samples = len(signal)
		if n_samples==0:
			return np.array([]), 0.0	

		_fft_complex = np.fft.rfft(signal) #Real FFT(양의주파수 성분만 반환)
		_fft_magnitude = np.abs(_fft_complex)/ n_samples
		_center_freq_Mhz = 0.0

		if fft_freqs_Mhz_in is not None and len(_fft_magnitude)>0:
			_peak_idx = np.argmax(_fft_magnitude)
			_center_freq_Mhz = fft_freqs_Mhz_in[_peak_idx]
			
		return _fft_magnitude, _center_freq_Mhz
		
	def apply_bandpass_filter(self, sampling_rate_in : float, lowcut_Mhz_in : float, highcut_Mhz_in : float, order: Optional[int] = 2):
		"""2. Butterworth 밴드패스 필터링 적용 (원할 때 개별 재실행 가능)"""
		if len(self.raw_data)==0:
			return
		nyquist = 0.5 * sampling_rate_in	
		raw_low = (lowcut_Mhz_in * 1e6) / nyquist
		raw_high = (highcut_Mhz_in * 1e6) / nyquist
		
		low = max(0.001, min(raw_low, 0.98))
		high = max(0.002, min(raw_high,0.99))
			
		if low >= high:
			high = min(low + 0.01, 0.99)
			
		b, a = butter(order, [low, high],btype='band')
		self.filtered_data =filtfilt(b, a, self.raw_data)	
		pass
			
	def extract_envelope(self, signal:np.ndarray) -> Optional[np.ndarray]:
		"""Hilbert 변환 기반 Envelope 추출 함수 (주파수 입력 불필요)"""
		if len(signal) == 0 :
			return None
		_analytic_signal = hilbert(signal)
		return np.abs(_analytic_signal)

	def envelope_at_TGC_ROI(self) : 
		self.env_tgc_ROI_Max = np.abs(hilbert(self.tgc_ROI_Max))
		self.env_tgc_ROI_min = np.abs(hilbert(self.tgc_ROI_min))
		self.env_tgc_ROI_Corr = np.abs(hilbert(self.tgc_ROI_Corr))
		self.env_tgc_ROI_Envelope = np.abs(hilbert(self.tgc_ROI_Envelope))
		pass
								
	def Make_roi_buffer(self, signal : np.ndarray, pre :int, post: int) :
		_pre_samples = pre
		_post_samples = post
		_numbers_ROI = pre + post

		#초기화
		self.ROI_Envelope_Raw = np.zeros(_numbers_ROI)
		self.ROI_Max = np.zeros(_numbers_ROI)
		self.ROI_min = np.zeros(_numbers_ROI)
		self.ROI_Corr  = np.zeros(_numbers_ROI)

		_align_idx = self.align_indices.get('envelope_peak')
		self.copy_roi_buffer(_align_idx, _pre_samples, _post_samples, self.ROI_Envelope_Raw, signal)

		_align_idx = self.align_indices.get('pos_max')
		self.copy_roi_buffer(_align_idx, _pre_samples, _post_samples, self.ROI_Max, signal)

		_align_idx = self.align_indices.get('neg_min')
		self.copy_roi_buffer(_align_idx, _pre_samples, _post_samples, self.ROI_min, signal)

		_align_idx = self.align_indices.get('cross_corr')
		self.copy_roi_buffer(_align_idx, _pre_samples, _post_samples, self.ROI_Corr, signal)

	@staticmethod
	def apply_log_compression(data_in: np.ndarray, k : float = 0.005, dynamic_range_db : float = 40.0) -> np.ndarray :
	    
        #[메모리 최적화] Log Compression 처리 후 바로 8-bit 정수(np.uint8, 0~255)로 변환하여 반환
       
		#data_in : 양수 형태 int 배열
		#k :로그 곡선의 기울기 조절하는 파라미터
		#dynamic_range_db : 최대 진폭 대비 잘라낼, 다이나믹 래인지 상한선. 하한 노이즈를 절단 Cliping하여 대비 명확히

		
		# [1 단계] 입력 데이터 안전성 확보 (음수값이나 0에 의한 np.log10 에러 방지)
		data_safe = np.maximum(data_in, 0.0)
		# [2 단계] 기본 Log Compression 계산: S_out = 20 * log10(1 + k * S_in)
		data_log = 20.0 * np.log10(1.0+ k * data_safe)
		#k가 커질 때 (예: $k = 0.1$ ~ $1.0$): 로그 곡선이 매우 급격히 꺾입니다. 작은 신호(내부 결함, 노이즈)를 폭발적으로 끌어올려 밝게 만들고, 큰 신호는 강하게 누릅니다.
		#k가 작아질 때 (예: $k = 0.0001$ ~ $0.001$): 로그 곡선이 직선(Linear)에 가까워집니다. 미세 신호 뻥튀기 효과가 줄어들고 큰 신호 위주로 선명해집니다.

		# [3 단계] Dynamic Range Clipping (선택 옵션) : 
		# 특정 dB 이하의 미세 노이즈를 0으로 자르고 상한선을 제한
		#최대 피크 신호 대비 몇 dB까지의 신호만 화면에 보여줄 것인가
		#dynamic_range_db = 40.0으로 설정하면, 최고 신호보다 40dB 이상 작은 노이즈 및 자잘한 바닥 신호들은 완전히 검은색(0) 처리
		#Log Compression으로 인해 함께 올라온 배경 바닥 노이즈를 깔끔하게 잘라내어(Clipping) 이미지의 대비(Contrast)를 또렷하게 만드는 역할
		if dynamic_range_db is not None:
			max_val = np.max(data_log)
			min_cutoff = max_val - dynamic_range_db
			# cutoff 미만의 미세 노이즈는 cutoff 값으로 바닥을 맞 춤
			#정해둔 최소값(min)보다 작은 값은 최소값으로, 최대값(max)보다 큰 값은 최대값으로
			data_log = np.clip(data_log, min_cutoff, max_val)

		# [4 단계] [0.0, 1.0] 범위 정규화 (Min-Max Normalization)
		min_v = np.min(data_log)
		max_v = np.max(data_log)

		# 분모가 0이 되는 Zero-division 방지
		if max_v - min_v > 1e-12:
			# ($0.000000000001$)라는 0에 매우 가까운 극소값
			# "최대값과 최소값의 차이가 0이 아니라 실제로 값이 존재하는가?
			data_norm = (data_log - min_v) / (max_v - min_v)
			#로그 압축이 완료된 임의의 범위 수치들을 최종 0.0 ~ 1.0 (Float) 표준 범위로 맞춥니다.
		else : 
			data_norm = np.zeros_like(data_log)
			#기존에 존재하는 어떤 배열(Array)의 '모양(Shape)'과 '데이터 타입(dtype)'을 그대로 복사해서, 값만 전부 0으로 채워진 새 배열
		
		#return : 0~1범위로 정규화 된 Log 압축 데이터 => np.ndarray바로 8-bit unsigned integer (0~255)로 캐스팅하여 메모리 절감
		return (data_norm * 255.0).astype(np.uint8)

	def copy_roi_buffer(self, align_idx : int, pre_samples : int, post_samples : int, ROI : np.ndarray, signal : np.ndarray) :
		_raw_start = align_idx - pre_samples
		_raw_end = align_idx + post_samples
		sig_start = max(0, _raw_start)
		sig_end = min(len(signal), _raw_end)
		roi_start = sig_start - _raw_start #항상 0 이상
		roi_end = roi_start + (sig_end - sig_start)
		ROI[roi_start:roi_end] = signal[sig_start:sig_end]	
					
		
	def process_full_pipeline(self,
							sampling_rate:float, 
							lowcut_Mhz:float, highcut_Mhz:float, order:int, 
							pre_align : int, post_align : int,
							fft_freqs_Mhz_in : Optional[np.ndarray] = None, 
							ref_template:Optional[np.ndarray]=None,
							tgc_gain_ndarray :Optional[np.ndarray]=None):
		#최초 1회 사전 계산

		#1. Raw Analysis 
		self.fft_magnitude, self.center_freq_Mhz = self.compute_fft(self.raw_data, fft_freqs_Mhz_in)
		self.raw_envelope_data = self.extract_envelope(self.raw_data)

		#2. BandPass Analysis
		self.apply_bandpass_filter(sampling_rate,lowcut_Mhz,highcut_Mhz,order)

		#3. Filtered 파형 FFT, Envelope : 기본 생성이네
		if self.filtered_data is not None:
			self.filtered_fft_magnitude, self.filtered_center_freq_Mhz = self.compute_fft(self.filtered_data, fft_freqs_Mhz_in)
			self.filtered_envelope_data = self.extract_envelope(self.filtered_data)

		#4. Align
		self.compute_all_align_indices(ref_template = ref_template)

		#5. ROI생성
		self.Make_roi_buffer(signal = self.filtered_data, pre = pre_align, post = post_align)

		#5. TGC 보정 적용 : Filtered 신호 기반	 
		self.apply_tgc_all(tgc_gain_ndarray = tgc_gain_ndarray)

		#6 TGC된 ROI에 Envelope 적용
		self.envelope_at_TGC_ROI()

		#7. B Scan용 8bit Log Compression 데이터 생성 (메모리 절감)
		self.roi_bscan_bytes = AScan.apply_log_compression(self.env_tgc_ROI_Envelope)


		
#CSV 파일 로더 클래스
class CSVReader:
	#csv 파일 읽어서, AScan객체 리스트 생성
	def __init__(self, config_in : ExperimentConfig):
		self.config = config_in
		self.tgc_gain_linear_ROI : Optional[np.ndarray] = None

	def tgc_setting(self):
		indices = np.arange(self.config.align_pre_samples + self.config.align_post_samples)
		#start_sample 이후부터 거리 비례 dB 증폭 적용
		depth_offset = np.maximum(0, indices - self.config.tgc_start_sample_from_align)
		gain_dB = depth_offset * self.config.tgc_slope_dB
		self.tgc_gain_linear_ROI = 10.0 ** (gain_dB/20.0) #dB스케일을 선형 스케일로 변환
		pass

	def load_ref_template(self) -> Optional[np.ndarray] :
		#Config에 지정된 ref.csv 파일에서 첫 번째 행 template 데이터를 1회 로드
		_path = self.config.ref_template_csv_path
		if not _path or not os.path.exists(_path) :
			return None
		try:
			df = pd.read_csv(_path, header=None, nrows=1)
			return df.iloc[0].dropna().values.astype(float)
		except Exception as e:
			print(f"[Warning] Ref Template 로드 실패 : {e}")
			return None

	def load_file(self, file_path_in:str) -> tuple[Optional[List[AScan]],Optional[np.ndarray],Optional[np.ndarray]]:
		try:
			# csv 파일 데이터, ref 파형 데이터에서 처음 1회만 실행
			_ref_template = self.load_ref_template()

			#tgc 게인 Set
			self.tgc_setting()

			df = pd.read_csv(file_path_in,header=None)
			#파일 로드 시점 : 첫번째 Row 기반으로 공통 시간/주파수축 1회만 생성
			_first_row = df.iloc[0].dropna().values.astype(float)
			self.config.number_of_samples_in_Whole_Abeam = len(_first_row)
		
			#sample index : x축 공통
			_shared_sample_indices = np.arange(self.config.number_of_samples_in_Whole_Abeam, dtype = int)
			_time_distance = 1.0 / self.config.sampling_rate
			_freqs_hz = np.fft.rfftfreq(self.config.number_of_samples_in_Whole_Abeam, d = _time_distance)
			#Real FFT 실행했을 때 각 주파수 성분이 몇 Hz(헤르츠)인지 물리적 주파수 축을 계산해 주는 NumPy 함수
			#Hz보다 MHz 표기가 훨씬 직관적이기 때문
			_shared_fft_freqs_Mhz = _freqs_hz/ 1e6
			
			ascan_list=[]
			for idx, row in df.iterrows():
				#Nan제거 및 float 변환
				signal_array = row.dropna().values.astype(float)

				if(len(signal_array)) == 0:
					raise ValueError("해당 Row에 맞는 data가 없습니다.")

				ascan = AScan(raw_data_in=signal_array,
					row_index_in = 0,# Row = 0 고정
					col_index_in = int(idx))# Col = CSV 내 행 번호
				ascan.process_full_pipeline(
					sampling_rate = self.config.sampling_rate,
					lowcut_Mhz = self.config.filter_lowcut_Mhz,
					highcut_Mhz = self.config.filter_highcut_Mhz,
					order = self.config.filter_order,
					fft_freqs_Mhz_in = _shared_fft_freqs_Mhz,
					ref_template = _ref_template,
					tgc_gain_ndarray = self.tgc_gain_linear_ROI,
					pre_align=self.config.align_pre_samples,
					post_align=self.config.align_post_samples)
				ascan_list.append(ascan)

			if len(ascan_list)==0:
				raise ValueError("None A Beam imported")
			
			return ascan_list, _shared_sample_indices, _shared_fft_freqs_Mhz #자동으로 튜플 묶임
			
		except Exception as e:
			raise RuntimeError(f"csv 파일을 읽는 중 에러가 발생 : {str(e)}")
			
#4. AScanViewerGUI : 화면 UI 클래스
class AScanViewerGUI:
	def __init__(self):

		#window 배율 150% 방지
		try:
			#ctypes.windll.shcore.SetProcessDpiAwareness(1)
			pass
		except Exception : 
			pass
		
		self.window = tk.Tk()
		self.window.title("Ultrasound Signal Processor : STEP4 (Align & ROI)")
		self.window.geometry("1280x720")
		
		#데이터 분석 객체 생성
		self.config = ExperimentConfig()
		self.csv_reader = CSVReader(self.config)
		self.ascan_list : List[AScan] = [] #csv에서 불러온 ascan 객체들이 들어갈 리스트
		self.current_ascan : Optional[AScan] = None #현재 화면에출력중인 Ascan객체
		self.total_cols : int = 0
		self.roi_x = np.arange(-self.config.align_pre_samples, self.config.align_post_samples)
		self.roi_y_signal = np.zeros(self.config.align_pre_samples + self.config.align_post_samples, dtype = float)
		self.roi_y_env = np.zeros(self.config.align_pre_samples + self.config.align_post_samples, dtype = float)
		#Align 예외 처리까지 고려한 padding
				
		#공통 x 축 변수 : 샘플 인덱스 배열
		self.shared_sample_indices : Optional[np.ndarray] = None
		self.shared_fft_freqs_Mhz:Optional[np.ndarray] = None

		#UI 컨트롤 변수들 : 뷰 모드 선택 : 라디오 변수(raw/filtered), Align
		self.view_mode_var = tk.StringVar(value="raw")
		self.align_method_var = tk.StringVar(value = self.config.align_method)
		
		# Matplotlib Line 객체 참조 변수 (고속 데이터 업데이트용)
		self.line_whole_signal : Optional[Line2D] = None #파형:Raw / Filtered
		self.line_whole_env : Optional[Line2D] = None #Envelope :모두 존재
		self.line_align_marker : Optional[Line2D] = None #동적 수직선 마커 저장용

		self.line_whole_fft : Optional[Line2D] = None #FFT:모두 존재
		self.line_peak_freq: Optional[Line2D] = None# 🎯 초록색 Peak 주파수 수직 가이드라인 추가

		self.line_roi_signal : Optional[Line2D] = None #ROI 파형
		self.line_roi_env : Optional[Line2D] = None #ROI 엔벨롭
	
		#화면 구성폼(버튼, 그래프)생성
		self.create_widgets()
		
	def create_widgets(self):

		#-------------------------------------
		# 상단  Control Panel : 버튼 및 설정 영역
		#-------------------------------------

		_control_frame = ttk.LabelFrame(self.window,text="Control Panel",padding=8)
		_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5) #x여백좌우:10px,y여백위아래:5px

		#좌상단 1열: csv 파일 열기 버튼
		_btn_open=ttk.Button(_control_frame,text="Open CSV",command=self.open_csv)
		_btn_open.grid(row=0, column=0, padx=5, pady=2)

		self.lbl_status=ttk.Label(_control_frame,text="No CSV File",font=("Consolas",9,"italic"),foreground="gray")
		self.lbl_status.grid(row=0,column=1,padx=10,pady=2,sticky="w")#왼쪽정렬

		#좌상단 2열 : Col 번호 선택 스핀박스(위/아래 화살표 버튼으로 숫자 바꾸기)
		ttk.Label(_control_frame,text="Col Index:").grid(row=0,column=2,padx=(15,5),pady=2)
		self.spin_col = ttk.Spinbox(_control_frame, from_=0, to=0, width=8, command=self.on_col_change)
		self.spin_col.grid(row=0, column=3, padx=5, pady=2)
		self.spin_col.bind("<Return>", self._on_enter_pressed)
		
		#좌상단 3열 : 토글 (Raw View vs Filtered View)
		ttk.Label(_control_frame, text="Display Mode:").grid(row=1,column=0,padx=(15,5),pady=2)
		_low_Mhz, _high_Mhz=int(self.config.filter_lowcut_Mhz),int(self.config.filter_highcut_Mhz)
		_filtered_btn_text = f"Filtered {_low_Mhz}-{_high_Mhz}Mhz"

		ttk.Radiobutton(_control_frame,text=_filtered_btn_text, variable= self.view_mode_var, value="filtered",command=self.on_view_mode_change).grid(row=1,column=1,padx=3,pady=2)
		ttk.Radiobutton(_control_frame,text="Raw Data", variable= self.view_mode_var,  value="raw", command=self.on_view_mode_change).grid(row=1,column=2,padx=3,pady=2)
		#ttk.Radiobutton(_control_frame,text="TGC Signal", variable=self.view_mode_var, value="tgc", command=self.on_view_mode_change).grid(row=1,column=3,padx=3,pady=2)

		#좌상단 4열 : align 라디오 버튼 4종
		ttk.Separator(_control_frame, orient='vertical').grid(row=0,column=4,rowspan=2,sticky="ns",padx=15) #구간 나누기
		ttk.Label(_control_frame, text="Align Method:",font=("Segoe UI",9,"bold")).grid(row=0,column=5,rowspan=2,padx=(0,5),pady=2)

		align_methods=[
			("Envelope Peak","envelope_peak",0,6),
			("Pos Max","pos_max",0,7),
			("Neg Min","neg_min",1,6),
			("Cross corr","cross_corr",1,7),
		]

		for text, val, r, c in align_methods:
			ttk.Radiobutton(
				_control_frame,
				variable = self.align_method_var,
				text=text,
				value=val,
				command=self.on_align_method_change
			).grid(row=r, column=c, padx=6, pady=2, sticky='w')#좌측정렬

		#-------------------------------------
		# 하단  그래프 출력 영역 plot panel
		#-------------------------------------
		_plot_frame = ttk.Frame(self.window)
		_plot_frame.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=10,pady=5)
		
		#Matplotlib 도화지 Figufre 생성
		self.fig = matplotlib.figure.Figure(figsize=(12, 4.8), dpi=100)#액자 1200px,600px
		gs = GridSpec(2,2, figure = self.fig, width_ratios=[1,1.2])#오른쪽열 가로폭, 왼쪽열보다 1.2배 넓게 설정
		self.ax_whole = self.fig.add_subplot(gs[0,0])#좌상단
		self.ax_fft = self.fig.add_subplot(gs[1,0])#좌하단
		self.ax_roi = self.fig.add_subplot(gs[:,1])#우측전체

		self.set_plot_style()
		
		#이를 Tkinter 창 내부에 집어넣기 위해 FigureCanvasTkAgg라는 '연결 다리(도화지)'
		self.canvas=FigureCanvasTkAgg(figure = self.fig, master = _plot_frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

		#Matpolotlib 툴바(확대,이동,저장버튼)추가
		_toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas,window=_plot_frame)
		_toolbar.update()

		
	def set_plot_style(self):
		#초기 1회만 : 축, 격자, 빈 선 Line 셋팅

		#1 : 좌상단 Raw Signal 그래프 셋팅
		self.ax_whole.set_title("Whole Signal",fontsize=9,fontweight='bold')
		self.ax_whole.set_xlabel("Index", fontsize=7)
		self.ax_whole.set_ylabel("Amplitude",fontsize=7)
		self.ax_whole.set_ylim(-32768,32768)
		self.ax_whole.grid(True,linestyle='--',alpha=0.5)
			
		# Line 객체 최초 단 1회 생성(고속 업데이트용 참조 보유)
		self.line_whole_signal = self.ax_whole.plot([],[],color='#1f77b4',linewidth=0.8,label="Signal")[0]
		self.line_whole_env = self.ax_whole.plot([],[],color='#ff7f0e', linewidth=1.2, linestyle='--', label="Envelope")[0]
		self.ax_whole.legend(loc='upper right', fontsize = 7)
		self.line_align_marker = self.ax_whole.axvline(x=0, color='black', linestyle=':', linewidth=1, visible=True)
		self.line_inverse_marker = self.ax_whole.axvline(x=0, color='purple', linestyle=':', linewidth=1, visible=False)
			
		#2 : 좌하단 FFT Spectrum 그래프 셋팅
		self.ax_fft.set_title("FFT Frequency Domain",fontsize=9,fontweight='bold')
		self.ax_fft.set_xlabel("Frequency Mhz", fontsize=7)
		self.ax_fft.set_ylabel("Magnitude",fontsize=7)
		self.ax_fft.set_xlim(0,100)# 초음파 주파수 대역인 0~100MHz 범위 표시
		self.ax_fft.grid(True,linestyle='--',alpha=0.5) 

		self.line_whole_fft = self.ax_fft.plot([],[],color='#d62728',linewidth=1.0)[0]
		self.line_peak_freq = self.ax_fft.axvline(x=0,color='#2ca02c',linestyle='--',linewidth=1.5,alpha=0.85,label='Peak Freq')

		#3. ROI Detail Chart
		self.ax_roi.set_title("ROI Align",fontsize=10,fontweight='bold')
		self.ax_roi.set_xlabel('Index',fontsize=8)
		self.ax_roi.set_ylabel('Ampltitude',fontsize=8)
		self.ax_roi.set_ylim(-32768,32768)
		self.ax_roi.grid(True, linestyle='--', alpha=0.5)
		self.ax_roi.set_xlim(-self.config.align_pre_samples,self.config.align_post_samples)

		#ROI Line객체도 단 1회 생성 및 보관(포인터 재활용)
		self.line_roi_signal = self.ax_roi.plot([],[],color='#1f77b4',linewidth=1.0,label="Signal")[0]
		self.line_roi_env = self.ax_roi.plot([], [], color='#ff7f0e', linewidth=1.2, linestyle='--', label="Envelope")[0] #label과 칼라가 다르네
		self.ax_roi.legend(loc='upper right',fontsize=6)


		self.fig.tight_layout()
		
	def open_csv(self):
		"""[Open CSV] 버튼 클릭시 실행 함수"""
		_file_path=filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
		if not _file_path: # 파이썬에서는 값이 None, 빈 문자열(""), 숫자 0, 빈 리스트([]) 등일 때 이를 False(거짓)
			return
		try:
			#csv 데이터 읽어오기
			self.ascan_list, self.shared_sample_indices, self.shared_fft_freqs_Mhz = self.csv_reader.load_file(_file_path)#자동으로 언패킹
			self.total_cols = len(self.ascan_list)
		
			#UI 상태 업데이트(파일명 표시 및 스핀박스 범위 설정)
			_filename= _file_path.split("/")[-1] #마지막 원소
			self.lbl_status.config(text=f"Loaded:{_filename}({self.total_cols} cols)",foreground="green")
			self.spin_col.config(from_=0, to = self.total_cols - 1)#규칙바꾸기
			self.spin_col.delete(0, tk.END)#기존 스핀박스 남아있던 이전번호(예:150) 지우는역할
			self.spin_col.insert(0,"0") #초기화

			# 최적화 : Matplotlib Line에 공통 X축 단 1회 고정
			self.line_whole_signal.set_xdata(self.shared_sample_indices)
			self.line_whole_env.set_xdata(self.shared_sample_indices)
			self.ax_whole.set_xlim(self.shared_sample_indices[0], self.shared_sample_indices[-1])	
			self.line_whole_fft.set_xdata(self.shared_fft_freqs_Mhz)

			self.line_roi_signal.set_xdata(self.roi_x)
			self.line_roi_env.set_xdata(self.roi_x)

			# FFT Y축 스케일 조절 : n sample -> fft y limit auto calibration
			_est_peak = (32768.0 * 90.0) / len(self.shared_sample_indices)
			_dynamic_fft_ylim = math.floor(_est_peak / 100.0) * 100 #100 단위 내림
			_fft_ylim_max = max(100, _dynamic_fft_ylim - 100)# 최소 100보장
			self.ax_fft.set_ylim(0, _fft_ylim_max)  # FFT Y축도 절대 기준으로 고정!
	
			#첫번째 0번 Row그래프 출력
			self.display_ascan(col_index_in = 0)
			
		except Exception as e:
			messagebox.showerror("Error",f"Failed to  load CSV file:\n{str(e)}")
			
	def _on_enter_pressed(self, event_L):
		self.on_col_change()
		
	def on_col_change(self):
		"""스핀박스 숫자 변경시 자동 실행"""
		if self.ascan_list is None :
			return
		try:
			_col_idx = int(self.spin_col.get())
			if 0 <= _col_idx < len(self.ascan_list):
				self.display_ascan(col_index_in = _col_idx)
			else:
				messagebox.showerror("Warning",f"Index out of Range(0~{len(self.ascan_list)-1})")
		except ValueError as e:
			messagebox.showerror("Error", e.__str__())
			
	def display_ascan(self, col_index_in : int):
		self.current_ascan = self.ascan_list[col_index_in]
		self.update_plots()
		self.canvas.draw()
		
	def on_view_mode_change(self):
		#모드 토글시 화면 즉시 전환
		if self.current_ascan is not None:
			self.update_plots()
			self.canvas.draw()

	def on_align_method_change(self):
		self.config.align_method = self.align_method_var.get()

		if self.current_ascan is not None:
			self.update_plots()
			self.canvas.draw()

	def update_roi_buffer(self, signal_in : np.ndarray, env_in : np.ndarray, align_idx : int) :
		'''기존 buffer를 그대로 재사용'''
		#초기화
		self.roi_y_signal[:] = 0.0
		self.roi_y_env[:]  = 0.0

		raw_start = align_idx - self.config.align_pre_samples
		raw_end = align_idx + self.config.align_post_samples

		sig_start = max(0, raw_start)
		sig_end = min(len(signal_in), raw_end)

		roi_start = sig_start - raw_start #항상 0 이상
		roi_end = roi_start + (sig_end - sig_start)

		self.roi_y_signal[roi_start:roi_end] = signal_in[sig_start:sig_end]	
		self.roi_y_env[roi_start:roi_end] = env_in[sig_start:sig_end]
					
	def update_plots(self):		
		#토글에 따른 y축 배열 바인딩만수행 (속도 극대화)
		_mode = self.view_mode_var.get()
		#_col_idx = self.current_ascan.col_index
		_align = self.align_method_var.get()

		if _mode == 'raw':
			_sig_data = self.current_ascan.raw_data
			_env_data = self.current_ascan.raw_envelope_data
			_fft_data = self.current_ascan.fft_magnitude
			_peak_freq = self.current_ascan.center_freq_Mhz
			self.ax_whole.set_title(f"1.Raw AScan Signal",fontsize=8, fontweight='bold')
			self.ax_fft.set_title(f"2.Raw FFT : Peak={_peak_freq:.1f}Mhz")

		else: # "filtered"
			_sig_data = self.current_ascan.filtered_data
			_env_data = self.current_ascan.filtered_envelope_data
			_fft_data = self.current_ascan.filtered_fft_magnitude
			_peak_freq = self.current_ascan.filtered_center_freq_Mhz	
			self.ax_whole.set_title(f"1.Filtered AScan Signal",fontsize=8, fontweight='bold')
			self.ax_fft.set_title(f"2.Filtered FFT : Peak={_peak_freq:.1f}Mhz")

		#1. whole AScan 업데이트
		self.line_whole_signal.set_ydata(_sig_data)
		self.line_whole_env.set_ydata(_env_data)
		
		#2. FFT
		self.line_whole_fft.set_ydata(_fft_data)
		self.line_peak_freq.set_xdata([_peak_freq, _peak_freq])
	
		align_idx = self.current_ascan.get_align_index(self.config.align_method)

		if align_idx == None : 
			align_idx = 0
			self.ax_roi.set_title(f"ROI Align : {None}")
		else :
			if self.current_ascan.phase_inverse != None :
				self.line_inverse_marker.set_xdata([self.current_ascan.phase_inverse, self.current_ascan.phase_inverse])
				self.line_inverse_marker.set_visible(True)
				self.ax_roi.set_title(f"ROI Align {align_idx}, Inverse {self.current_ascan.phase_inverse}")
			else:
				self.line_inverse_marker.set_visible(False)
				self.ax_roi.set_title(f"ROI Align {align_idx}")

		self.line_align_marker.set_xdata([align_idx, align_idx])

		#3. ROI 파형  업데이트
		if _align == "envelope_peak":
			self.roi_y_signal = self.current_ascan.tgc_ROI_Envelope
			self.roi_y_env = self.current_ascan.env_tgc_ROI_Envelope
		elif _align == "pos_max":
			self.roi_y_signal = self.current_ascan.tgc_ROI_Max
			self.roi_y_env = self.current_ascan.env_tgc_ROI_Max
		elif _align == "neg_min":
			self.roi_y_signal = self.current_ascan.tgc_ROI_min
			self.roi_y_env = self.current_ascan.env_tgc_ROI_min
		elif _align == "cross_corr":
			self.roi_y_signal = self.current_ascan.tgc_ROI_Corr
			self.roi_y_env = self.current_ascan.env_tgc_ROI_Corr
					
		#4. Line 포인터 교체	
		self.line_roi_signal.set_ydata(self.roi_y_signal)
		self.line_roi_env.set_ydata(self.roi_y_env)

	def run(self):
		self.window.mainloop()

#5. AScan 리스트로부터 메모리 효율적인 RGB 3차원 B-Scan 이미지를 생성하는 클래스

class BScanProcessor : 
	@staticmethod
	def generate_bscan_rgb(ascan_list : List[AScan], enable_inverse_overlay : bool = False) -> np.ndarray :

		if ascan_list is None or ascan_list[0].roi_bscan_bytes is None :
			return np.array([])

		num_cols = len(ascan_list)
		roi_len = len(ascan_list[0].roi_bscan_bytes)

		#1차원 greyScale uint8 행렬 구축 (Nz, Nx)
		#각 AScan의 8bit uint8 데이터를 열(collumn) 단위로 쌓음
		bscan_gray = np.zeros((roi_len,num_cols), dtype=np.uint8)
		



#실행부
if __name__ == "__main__":
	#app객체 생성
	app = AScanViewerGUI()
	app.run()