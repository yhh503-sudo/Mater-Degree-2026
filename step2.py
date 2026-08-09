import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import math

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

#SciPy 신호처리 모듈
from scipy.signal import butter, hilbert, filtfilt

# 실험 장비 및 환경 설정 관리 클래스
@dataclass
class ExperimentConfig :
	sampling_rate:float = 1e9 #샘플링 속도(기본: 1 GHz = 1,000,000,000 Hz)
	probe_center_freq:float=45e6 #(기본: 45 MHz)
	bit_depth:int=15 # ADC Data Range (+- 2^15)
	
	#SETP3 : 필터링 기본 설정값(Mhz)
	filter_lowcut_mhz : float = 20.0 #DC 오프셋 및 아주 낮은 진동 노이즈 제거
	filter_highcut_mhz : float = 70.0#초고주파 백그라운드 노이즈만 제거
	filter_order : int= 2 #차수(Order)를 4 -> 2로 낮추면 천이 경계가 더 완만해져 원 신호 변형 최소화
	
#AScan:단일 AScan 데이터 클래스
class AScan:
	def __init__(self, raw_data_in:np.ndarray, row_index_in:int=0, col_index_in:int=0 ):
		self.raw_data :np.ndarray = raw_data_in
		#변수 이름 뒤에 콜론(:) 타입을 적어주는 것은 "이 변수에는 이러한 타입의 데이터가 들어올 예정입니다"라고 사람(개발자)과 에디터(VS Code)에게 알려주는 설명서(주석)
		self.row_index : int = row_index_in# 단일 CSV 시 0으로 고정
		self.col_index : int = col_index_in# CSV 파일 내 행(Row) 번호
		
		# 1. Raw 파형 분석 결과
		self.fft_magnitude:Optional[np.ndarray]=None #set_ydata(self.current_ascan.fft_magnitude)
		self.center_freq_mhz:Optional[float]=None
		self.raw_envelope_data:Optional[np.ndarray]=None
		
		#2. Filtered 파형 분석 결과
		self.filtered_data:Optional[np.ndarray]=None
		self.filtered_fft_magnitude:Optional[np.ndarray]=None
		self.filtered_center_freq_mhz:Optional[np.ndarray]=None
		self.filtered_envelope_data:Optional[np.ndarray]=None
	
	# ==========================================
    # 🎯 독립된 개별 신호 처리 메서드 모음
    # ==========================================	
	def compute_fft(self, signal: np.ndarray, freqs_mhz:Optional[np.ndarray]=None) -> Tuple[Optional[np.ndarray],Optional[float]]:
		"""통일 FFT 함수 (Magnitude 배열, Center Frequency 튜플 반환)"""
		n_samples = len(signal)
		if n_samples==0:
			return None, None	

		_fft_complex = np.fft.rfft(signal) #Real FFT(양의주파수 성분만 반환)
		_fft_magnitude = np.abs(_fft_complex)/ n_samples
		
		_center_freq_mhz=None
		if freqs_mhz is not None and len(_fft_magnitude)>0:
			_peak_idx = np.argmax(_fft_magnitude)
			_center_freq_mhz = freqs_mhz[_peak_idx]
			
		return _fft_magnitude, _center_freq_mhz
		
	def apply_bandpass_filter(self, sampling_rate_in : float, lowcut_mhz_in : float, highcut_mhz_in : float, order: Optional[int] = 2):
		"""2. Butterworth 밴드패스 필터링 적용 (원할 때 개별 재실행 가능)"""
		if len(self.raw_data)==0:
			return
		nyquist = 0.5 * sampling_rate_in	
		raw_low = (lowcut_mhz_in * 1e6) / nyquist
		raw_high = (highcut_mhz_in * 1e6) / nyquist
		
		low = max(0.001, min(raw_low, 0.98))
		high = max(0.002, min(raw_high,0.99))
			
		if low >= high:
			high = min(low + 0.01, 0.99)
			
		b, a = butter(order, [low,high],btype='band')
		self.filtered_data =filtfilt(b,a,self.raw_data)	
			
	def extract_envelope(self, signal:np.ndarray)-> Optional[np.ndarray]:
		"""Hilbert 변환 기반 Envelope 추출 함수 (주파수 입력 불필요)"""
		if len(signal) ==0:
			return None
		_analytic_signal = hilbert(signal)
		return np.abs(_analytic_signal)
		
	def process_full_pipeline(self, sampling_rate:float, lowcut_mhz:float, highcut_mhz:float, order:int=2, freqs_mhz_in:Optional[np.ndarray]=None):
		#최초 1회 사전 계산
		#Raw
		self.fft_magnitude,self.center_freq_mhz = self.compute_fft(self.raw_data,freqs_mhz_in)
		self.raw_envelope_data = self.extract_envelope(self.raw_data)
		#BandPass
		self.apply_bandpass_filter(sampling_rate,lowcut_mhz,highcut_mhz,order)
		#Filtered 파형 FFT, Envelope
		if self.filtered_data is not None:
			self.filtered_fft_magnitude, self.filtered_center_freq_mhz = self.compute_fft(self.filtered_data, freqs_mhz_in)
			self.filtered_envelope_data = self.extract_envelope(self.filtered_data)
			
#CSV 파일 로더 클래스
class CSVReader:
	#csv 파일 읽어서, AScan객체 리스트 생성
	def __init__(self, config_in:ExperimentConfig):
		self.config=config_in
	def load_file(self,file_path_in:str) -> tuple[List[AScan],np.ndarray,np.ndarray]:
		try:
			df = pd.read_csv(file_path_in,header=None)
			#파일 로드 시점 : 첫번째 Row 기반으로 공통 시간/주파수축 1회만 생성
			_first_row = df.iloc[0].dropna().values.astype(float)
			_n_samples=len(_first_row)


			#공통시간축 us
			_shared_time_us = np.arange(_n_samples)/self.config.sampling_rate*1e6
			_freqs_hz = np.fft.rfftfreq(_n_samples, d=1.0 / self.config.sampling_rate)#공통주파수:Hz->Mhz 공통 배열 1회 생성
			_shared_fft_freqs_mhz = _freqs_hz / 1e6
			
			ascan_list=[]
			for idx, row in df.iterrows():
				#Nan제거 및 float 변환
				signal_array = row.dropna().values.astype(float)
				ascan = AScan(raw_data_in=signal_array,
					row_index_in = 0,# Row = 0 고정
					col_index_in = idx)# Col = CSV 내 행 번호
				ascan.process_full_pipeline(
					sampling_rate = self.config.sampling_rate,
					lowcut_mhz=self.config.filter_lowcut_mhz,
					highcut_mhz=self.config.filter_highcut_mhz,
					order=self.config.filter_order,
					freqs_mhz_in =_shared_fft_freqs_mhz)					
				ascan_list.append(ascan)
			return ascan_list,_shared_time_us,_shared_fft_freqs_mhz #자동으로 튜플 묶임
			
		except Exception as e:
			raise RuntimeError(f"파일을 읽는 중 에러가 발생 : {str(e)}")
			
#4. AScanViewerGUI : 화면 UI 클래스
class AScanViewerGUI:
	def __init__(self):
		self.window = tk.Tk()
		self.window.title("Ultrasound Signal Processor - STEP 3 (BandPass Envelope)")
		self.window.geometry("1200x650")
		
		#데이터 분석 객체 생성
		self.config = ExperimentConfig()
		self.csv_reader = CSVReader(self.config)
		self.ascan_list : List[AScan] = []#csv에서 불러온 ascan 객체들이 들어갈 리스트
		self.current_ascan:Optional[np.ndarray]=None #현재 화면에출력중인 Ascan객체
		self.total_cols : int = 0
		
		#공통 x 축 변수
		self.shared_time_us:Optional[np.ndarray]=None
		self.shared_fft_freqs_mhz:Optional[np.ndarray]=None
		
		#뷰 모드 선택 : 라디오 변수(raw/filtered)
		self.view_mode_var = tk.StringVar(value="raw")
		
		# Matplotlib Line 객체 참조 변수 (고속 데이터 업데이트용)
		self.line_signal : Optional[Line2D] = None #파형:Raw / Filtered
		self.line_env : Optional[Line2D] = None #Envelope :모두 존재
		self.line_fft : Optional[Line2D] = None #FFT:모두 존재
		self.line_pear_freq: Optional[Line2D] = None# 🎯 초록색 Peak 주파수 수직 가이드라인 추가
		#화면 구성폼(버튼, 그래프)생성
		self.create_widgets()
		
	def create_widgets(self):
		
		#상단 버튼 및 설정 영역 Control Panel
		_control_frame = ttk.LabelFrame(self.window,text="Control Panel",padding=10)
		_control_frame.pack(side=tk.TOP,fill=tk.X,padx=10,pady=5) #x여백좌우:10px,y여백위아래:5px
		#csv 파일 열기 버튼
		#콜백 함수"지금 실행하는게 아니라,이벤트(일)터졌을때 실행해달라'나중에 불러(Call Back)'줄 함수를 등록
		_btn_open=ttk.Button(_control_frame,text="Open CSV",command=self.open_csv)
		_btn_open.grid(row=0,column=0,padx=5,pady=5)
		#파일 상태 표시 글자
		self.lbl_status=ttk.Label(_control_frame,text="No CSV File",font=("Consolas",9,"italic"),foreground="gray")
		self.lbl_status.grid(row=0,column=1,padx=10,pady=5,sticky="w")#왼쪽정렬
		#Col 번호 선택 스핀박스(위/아래 화살표 버튼으로 숫자 바꾸기)
		ttk.Label(_control_frame,text="Select Col Index:").grid(row=0,column=2,padx=(20,5),pady=5)
		self.spin_col = ttk.Spinbox(_control_frame,from_=0,to=0,width=8,command=self.on_col_change)
		#from_, to : tkinter 스핀박스 숫자 범위
		self.spin_col.grid(row=0, column=3, padx=5, pady=5)
		self.spin_col.bind("<Return>", self._on_enter_pressed)
		#묶기 : Return = 엔터키 : 이벤트 떄, 호출함수 : self._on_col_change
		
		# 🎯 파형 / FFT 모드 토글 (Raw View vs Filtered View)
		ttk.Label(_control_frame,text="Display Mode:").grid(row=0,column=4,padx=(20,5),pady=5)
			#Q.왜 라디오버튼이 2개?
		self.config.filter_lowcut_mhz, self.config.filter_highcut_mhz
		_low_Mhz, _high_Mhz=int(self.config.filter_lowcut_mhz),int(self.config.filter_highcut_mhz)
		_filtered_btn_text = f"Filtered Data {_low_Mhz}-{_high_Mhz}Mhz"
		ttk.Radiobutton(_control_frame,text=_filtered_btn_text, variable=self.view_mode_var, value="filtered",command=self.on_view_mode_change).grid(row=0,column=5,padx=5,pady=5)
		ttk.Radiobutton(_control_frame,text="Raw Data", variable=self.view_mode_var,  value="raw",command=self.on_view_mode_change).grid(row=0,column=6,padx=5,pady=5)
	
		#하단 그래프 출력 영역 plot panel
		_plot_frame = ttk.Frame(self.window)
		_plot_frame.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=10,pady=5)
		
		#Matplotlib 도화지 Figufre 생성
		self.fig=matplotlib.figure.Figure(figsize=(10,5),dpi=100)#액자
		
		#self.ax=self.fig.add_subplot(111)#액자 안에 들어간 실제 그림: (행, 열, 위치)"화면 전체를 하나의 통 그래프로 쓰겠다"
		self.ax_signal= self.fig.add_subplot(121)#좌:Time Domain
		self.ax_fft = self.fig.add_subplot(122)#우:Frequency Domain
		self.set_plot_style()
		
		#도화지를 Tkinter 창 안에 붙이기 : Matplotlib 혼자창 띄우는 도구
		#이를 Tkinter 창 내부에 집어넣기 위해 FigureCanvasTkAgg라는 '연결 다리(도화지)'
		self.canvas=FigureCanvasTkAgg(figure=self.fig, master=_plot_frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
		#Matpolotlib 툴바(확대,이동,저장버튼)추가
		_toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas,window=_plot_frame)
		_toolbar.update()
		
	def set_plot_style(self):
		#초기 1회만 : 축, 격자, 빈 선 Line 셋팅
		#1:좌측 Raw Signal 그래프 셋팅
		self.ax_signal.set_title("AScan Signal & Envelope",fontsize=11,fontweight='bold')
		self.ax_signal.set_xlabel("Time(us)", fontsize=9)
		self.ax_signal.set_ylabel("Amplitude",fontsize=9)
		self.ax_signal.set_ylim(-32768,32768)
		self.ax_signal.grid(True,linestyle='--',alpha=0.6)
		
		self.line_signal= self.ax_signal.plot([],[],color='#1f77b4',linewidth=0.8,label="Signal")[0]
		self.line_env=self.ax_signal.plot([],[],color='#ff7f0e', linewidth=1.2, linestyle='--', label="Envelope")[0]
		self.ax_signal.legend(loc='upper right', fontsize=8)
	
		#2:우측 FFT Spectrum 그래프 셋팅
		self.ax_fft.set_title("FFT Spectrum(Frequency Domain)",fontsize=11,fontweight='bold')
		self.ax_fft.set_xlabel("Frequency Mhz", fontsize=9)
		self.ax_fft.set_ylabel("Magnitude",fontsize=9)
		self.ax_fft.set_xlim(0,100)# 초음파 주파수 대역인 0~100MHz 범위 표시

		#self.ax_fft.set_ylim(0, _fft_ylim_max)  # 🎯 FFT Y축도 절대 기준으로 고정!
		self.ax_fft.grid(True,linestyle='--',alpha=0.6) 
		self.line_fft=self.ax_fft.plot([],[],color='#d62728',linewidth=1.0)[0]
		self.line_peak_freq = self.ax_fft.axvline(x=0,color='#2ca02c',linestyle='--',linewidth=1.5,alpha=0.85,label='Peak Freq')
		#
		self.fig.tight_layout()
		
	def open_csv(self):
		"""[Open CSV] 버튼 클릭시 실행 함수"""
		_file_path=filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
		if not _file_path: # 파이썬에서는 값이 None, 빈 문자열(""), 숫자 0, 빈 리스트([]) 등일 때 이를 False(거짓)
			return
		try:
			#csv 데이터 읽어오기
			self.ascan_list, self.shared_time_us, self.shared_fft_freqs_mhz = self.csv_reader.load_file(_file_path)#자동으로 언패킹
			self.total_cols = len(self.ascan_list)
		
			#UI 상태 업데이트(파일명 표시 및 스핀박스 범위 설정)
			_filename= _file_path.split("/")[-1] #마지막 원소
			self.lbl_status.config(text=f"Loaded:{_filename}({self.total_cols} cols)",foreground="green")
			self.spin_col.config(from_=0, to = self.total_cols - 1)#규칙바꾸기
			self.spin_col.delete(0,tk.END)#기존 스핀박스 남아있던 이전번호(예:150) 지우는역할
			self.spin_col.insert(0,"0") #초기화

			# 최적화 : Matplotlib Line에 공통 X축 단 1회 고정
			self.line_signal.set_xdata(self.shared_time_us)
			self.line_env.set_xdata(self.shared_time_us)
			self.ax_signal.set_xlim(self.shared_time_us[0], self.shared_time_us[-1])	
			self.line_fft.set_xdata(self.shared_fft_freqs_mhz)

			#n sample -> fft y limit auto calibration
			_est_peak = (32768.0*90.0)/ len(self.shared_time_us)
			_dynamic_fft_ylim = math.floor(_est_peak/100.0)*100 #100 단위 내림
			_fft_ylim_max=max(100,_dynamic_fft_ylim)# 최소 100보장
			self.ax_fft.set_ylim(0, _fft_ylim_max)  # FFT Y축도 절대 기준으로 고정!
	
			#첫번째 0번 Row그래프 출력
			self.display_ascan(col_index_in=0)
			
		except Exception as e:
			messagebox.showerror("Error",f"Failed to  load CSV file:\n{str(e)}")
			
	def _on_enter_pressed(self,event_L):
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
		except ValueError:
			pass
			
	def display_ascan(self,col_index_in : int):
		self.current_ascan = self.ascan_list[col_index_in]
		self.update_plots()
		self.canvas.draw()
		
	def on_view_mode_change(self):
		#모드 토글시 화면 즉시 전환
		if self.current_ascan is not None:
			self.update_plots()
			self.canvas.draw()
			
	def update_plots(self):
		#토글에 따른 y축 배열 바인딩만수행 (속도 극대화)
		_mode = self.view_mode_var.get()
		_col_idx = self.current_ascan.col_index
		
		if _mode=='raw':
			#Raw파형 Envelope
			self.line_signal.set_ydata(self.current_ascan.raw_data)
			self.line_signal.set_label("Raw Signal")
			
			#Raw Evnelope
			if self.current_ascan.raw_envelope_data is not None:
				self.line_env.set_ydata(self.current_ascan.raw_envelope_data)
				self.line_env.set_label("Raw Envelope")
			self.ax_signal.set_title(f"1. Raw AScan Signal (Col Index: {_col_idx})",fontsize=10,fontweight='bold')
		
			#Raw FFT
			if self.current_ascan.fft_magnitude is not None:
				self.line_fft.set_ydata(self.current_ascan.fft_magnitude)
				self.line_fft.set_label("Raw FFT")
				
				peak_freq = self.current_ascan.center_freq_mhz
				self.ax_fft.set_title(f"2. Raw FFT Spectrum : Peak {peak_freq:.2f}Mhz", fontsize=10, fontweight='bold')
				self.line_peak_freq.set_xdata([peak_freq, peak_freq])

		else: # "filtered"

			if self.current_ascan.filtered_data is not None:
				self.line_signal.set_ydata(self.current_ascan.filtered_data)
				self.line_signal.set_label("Filtered Signal")
				
				#filtered Evnelope
				if self.current_ascan.filtered_envelope_data is not None:
					self.line_env.set_ydata(self.current_ascan.filtered_envelope_data)
					self.line_env.set_label("Filtered Envelope")
				
				self.ax_signal.set_title(f"1. Filtered AScan Signal (ColIndex: {_col_idx})", fontsize=10, fontweight='bold')
				
			#filtered FFT
			if self.current_ascan.filtered_fft_magnitude is not None:
				self.line_fft.set_ydata(self.current_ascan.filtered_fft_magnitude)
				self.line_fft.set_label("Filtered FFT")
				
				peak_freq = self.current_ascan.filtered_center_freq_mhz
				self.line_peak_freq.set_xdata([peak_freq,peak_freq])
				self.ax_fft.set_title(f"2. Filtered FFT Spectrum : Peak {peak_freq:.2f}Mhz", fontsize=10, fontweight='bold')
		
		self.ax_signal.legend(loc='upper right', fontsize=8)
		self.ax_fft.legend(loc='upper right', fontsize=8)
				
	def run(self):
		self.window.mainloop()
#실행부
if __name__ == "__main__":
	#app객체 생성
	app = AScanViewerGUI()
	app.run()