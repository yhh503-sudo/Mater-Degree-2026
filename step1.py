import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 실험 장비 및 환경 설정 관리 클래스
@dataclass
class ExperimentConfig :
	sampling_rate:float = 1e9 #샘플링 속도(기본: 1 GHz = 1,000,000,000 Hz)
	probe_center_freq:float=45e6 #(기본: 45 MHz)
	bit_depth:int=15 # ADC Data Range (+- 2^15)
#AScan:단일 AScan 데이터 클래스
class AScan:
	def __init__(self,raw_data_in:np.ndarray,config_in:ExperimentConfig,row_index_in:int=0):
		self.raw_data :np.ndarray = raw_data_in
		#변수 이름 뒤에 콜론(:) 타입을 적어주는 것은 "이 변수에는 이러한 타입의 데이터가 들어올 예정입니다"라고 사람(개발자)과 에디터(VS Code)에게 알려주는 설명서(주석)
		self.config:ExperimentConfig=config_in
		self.row_index:int=row_index_in
		#시간축 계산 ms단위
		self.time_us:np.ndarray=np.arange(len(self.raw_data))/self.config.sampling_rate*1e6
		# FFT 관련 멤버 변수
		self.fft_freqs_mhz:Optional[np.ndarray]=None
		self.fft_magnitude:Optional[np.ndarray]=None#Optional means None혹은 np.ndarray이다.
		self.center_freq_mhz:Optional[float]=None
		self.compute_fft()
		# 향후 STEP 3 확장용 멤버 변수 (예약) ---
		self.filtered_data:Optional[np.ndarray]=None
		self.envelope_data:Optional[np.ndarray]=None
	def compute_fft(self):
		#1차원 실수 신호에 대한 fast fourier transform 수행
		n_samples = len(self.raw_data)
		if n_samples==0:
			return
		#1. Real FFT(양의주파수 성분만 반환)
		fft_complex = np.fft.rfft(self.raw_data)
		#2. 주파수 축 계산(Hz-> Mhz변환) : d=sampling intervals
		freqs_hz = np.fft.rfftfreq(n_samples,d=1.0/self.config.sampling_rate)
		#예시 freqs_hz = [0, 10000000, 25000000, 45000000, 100000000] # (단위: Hz)
		self.fft_freqs_mhz = freqs_hz/1e6
		#3.진폭 스펙트럼 Magnitude 계산 및 정규화
		self.fft_magnitude = np.abs(fft_complex)/ n_samples
		#4.최대 Peak 지점의 중심 주파수 추정
		peak_idx=np.argmax(self.fft_magnitude)
		self.center_freq_mhz = self.fft_freqs_mhz[peak_idx]
		
		
#CSV 파일 로더 클래스
class CSVReader:
	#csv 파일 읽어서, AScan객체 리스트 생성
	def __init__(self, config_in:ExperimentConfig):
		self.config=config_in
	def load_file(self,file_path_in:str) -> List[AScan]:
		try:
			df = pd.read_csv(file_path_in,header=None)
			ascan_list=[]
			for idx, row in df.iterrows():
				#Nan제거 및 float 변환
				signal_array = row.dropna().values.astype(float)
				ascan=AScan(raw_data_in=signal_array,config_in=self.config,row_index_in=idx)
				ascan_list.append(ascan)
			return ascan_list
		except Exception as e:
			raise RuntimeError(f"파일을 읽는 중 에러가 발생했습니다 : {str(e)}")
			
#4. AScanViewerGUI : 화면 UI 클래스
class AScanViewerGUI:
	def __init__(self):
		self.window = tk.Tk()
		self.window.title("Ultrasound Signal Processor - STEP2 (FFT Spectrum Viewer)")
		self.window.geometry("1200x650")
		#데이터 분석 객체 생성
		self.config = ExperimentConfig()
		self.csv_reader=CSVReader(self.config)
		self.ascan_list=[]#csv에서 불러온 ascan 객체들이 들어갈 리스트
		self.current_ascan:Optional[AScan]=None #현재 화면에출력중인 Ascan객체
		# Matplotlib Line 객체 참조 변수 (고속 데이터 업데이트용)
		self.line_raw = None
		self.line_fft = None
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
		#Row 번호 선택 스핀박스(위/아래 화살표 버튼으로 숫자 바꾸기)
		_lbl_row_index=ttk.Label(_control_frame,text="Select Row Index:").grid(row=0,column=2,padx=(20,5),pady=5)
		self.spin_row=ttk.Spinbox(_control_frame,from_=0,to=0,width=8,command=self.on_row_change)
		#from_, to : tkinter 스핀박스 숫자 범위
		self.spin_row.grid(row=0,column=3,padx=5,pady=5)
		self.spin_row.bind("<Return>", self._on_enter_pressed)
		#묶기
		#Return = 엔터키
		#발생하는 이벤트 떄, 호출함수 : self._on_row_change
		
		#하단 그래프 출력 영역 plot panel
		_plot_frame = ttk.Frame(self.window)
		_plot_frame.pack(side=tk.TOP,fill=tk.BOTH,expand=True,padx=10,pady=5)
		#Matplotlib 도화지 Figufre 생성
		self.fig=matplotlib.figure.Figure(figsize=(10,4.5),dpi=100)#액자
		#self.ax=self.fig.add_subplot(111)#액자 안에 들어간 실제 그림: (행, 열, 위치)"화면 전체를 하나의 통 그래프로 쓰겠다"
		self.ax_raw = self.fig.add_subplot(121)#좌:Time Domain
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
		# self.ax.clear()
		# self.ax.set_title("Raw A-Scan Signal",fontsize=12,fontweight='bold')
		# self.ax.set_xlabel("Time (us)",fontsize=10)
		# self.ax.set_ylabel("Amplitude (+- 2^15)",fontsize=10)
		# self.ax.set_ylim(-32768,32768)
		# self.ax.grid(True,linestyle='--',alpha=0.6)
		# _lines = self.ax.plot([],[],color='#1f77b4',linewidth=1.0)
		# self.line=_lines[0]
		#1:좌측 Raw Signal 그래프 셋팅
		self.ax_raw.set_title("Raw AScan Signal(Time Domain)",fontsize=11,fontweight='bold')
		self.ax_raw.set_xlabel("Time(us)", fontsize=9)
		self.ax_raw.set_ylabel("Amplitude (+- 2^15)",fontsize=9)
		self.ax_raw.set_ylim(-32768,32768)
		self.ax_raw.grid(True,linestyle='--',alpha=0.6)
		_lines_Raw = self.ax_raw.plot([],[],color='#1f77b4',linewidth=1.0)
		self.line_raw=_lines_Raw[0]
		#2:우측 FFT Spectrum 그래프 셋팅
		self.ax_fft.set_title("FFT Spectrum(Frequency Domain)",fontsize=11,fontweight='bold')
		self.ax_fft.set_xlabel("Frequency Mhz", fontsize=9)
		self.ax_fft.set_ylabel("Magnitude",fontsize=9)
		self.ax_fft.set_xlim(0,100)# 초음파 주파수 대역인 0~100MHz 범위 표시
		self.ax_fft.grid(True,linestyle='--',alpha=0.6)
		_lines_FFT = self.ax_fft.plot([],[],color='#d62728',linewidth=1.0)
		self.line_fft=_lines_FFT[0]
		#
		self.fig.tight_layout()
		
	def open_csv(self):
		"""[Open CSV] 버튼 클릭시 실행 함수"""
		_file_path=filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
		if not _file_path: # 파이썬에서는 값이 None, 빈 문자열(""), 숫자 0, 빈 리스트([]) 등일 때 이를 False(거짓)
			return
		try:
			#csv 데이터 읽어오기
			self.ascan_list = self.csv_reader.load_file(_file_path)
			_total_rows=len(self.ascan_list)
			#UI 상태 업데이트(파일명 표시 및 스핀박스 범위 설정)
			_filename= _file_path.split("/")[-1] #마지막 원소
			self.lbl_status.config(text=f"Loaded:{_filename}({_total_rows}rows)",foreground="green")
			self.spin_row.config(from_=0,to=_total_rows-1)#규칙바꾸기
			self.spin_row.delete(0,tk.END)#기존 스핀박스 남아있던 이전번호(예:150) 지우는역할
			self.spin_row.insert(0,"0") #초기화

			#최적화 코드 : 파일 로드시 0번 data의 time_us로 x축을 1번만 설정
			first_ascan = self.ascan_list[0]
			self.line_raw.set_xdata(first_ascan.time_us)
			self.ax_raw.set_xlim(first_ascan.time_us[0], first_ascan.time_us[-1])

			#첫번째 0번 Row그래프 출력
			self.display_ascan(0)
		except Exception as e:
			messagebox.showerror("Error",f"Failed to  load CSV file:\n{str(e)}")
	def _on_enter_pressed(self,event_L):
		self.on_row_change()
		
	def on_row_change(self):
		"""스핀박스 숫자 변경시 자동 실행"""
		if not self.ascan_list:
			return
		try:
			_row_idx = int(self.spin_row.get())
			if 0<= _row_idx < len(self.ascan_list):
				self.display_ascan(_row_idx)
			else:
				messagebox.showerror("Warning",f"Index out of Range(0~{len(self.ascan_list)-1})")
		except ValueError:
			pass
	def display_ascan(self, row_index_in):
		#선택된 번호의 AScan 파형을 화면에 그리는 함수
		self.current_ascan = self.ascan_list[row_index_in]
		#1.좌측 Raw파형 갱신
		self.line_raw.set_ydata(self.current_ascan.raw_data)
		self.ax_raw.set_title(f"Raw AScan. RowIndex:{row_index_in}",fontsize=12,fontweight='bold')
		#2.우측 FFT스펙트럼 데이터 갱신
		if self.current_ascan.fft_freqs_mhz is not None and self.current_ascan.fft_magnitude is not None:
			self.line_fft.set_ydata(self.current_ascan.fft_magnitude)
			_peak_freq = self.current_ascan.center_freq_mhz
			self.ax_fft.set_title(f"FFT Spectrum(Peak:{_peak_freq:.2f}Mhz",fontsize=11,fontweight='bold')
		#화면새로고침 :여기에서 실제 화면이 바뀜. tk 매개 활동
		self.canvas.draw()
	def run(self):
		self.window.mainloop()
#실행부
if __name__ == "__main__":
	#app객체 생성
	app = AScanViewerGUI()
	app.run()