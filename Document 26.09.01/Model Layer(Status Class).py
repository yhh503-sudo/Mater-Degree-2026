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
from matplotlib.lines import Line2D
import ctypes
import time
from scipy.signal import butter, hilbert, filtfilt


# ==========================================
# 0. 인프라 Layer (Publisher)
# ==========================================
class DataEventPublisher: 
    """데이터 및 Config/State 변경 이벤트를 구독자에게 알리는 이벤트 발행자 클래스"""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[], None]]] = {}

    def subscribe(self, event_type_in: str, callback_in: Callable[[], None]) -> None: 
        if event_type_in not in self._subscribers:
            self._subscribers[event_type_in] = []
        self._subscribers[event_type_in].append(callback_in)

    def notify(self, _event_type_in: str) -> None: 
        """등록된 구독자들에게 이벤트 발생 통보"""
        if _event_type_in in self._subscribers: 
            for _callback in self._subscribers[_event_type_in]:
                _callback()


# ==========================================
# 1. Model & State Layer (Data & State)
# ==========================================

# [통합] 애플리케이션의 모든 가변 상태 및 파라미터 설정을 단일 관리하는 클래스
@dataclass
class AppState:
    # 1. UI 활성 선택 상태
    active_row: int = 0  
    active_col: int = 0  
    active_align_map: Optional[np.ndarray] = field(default=None, init=False)

    # 2. 하드웨어 및 수집 환경 설정
    sampling_rate: float = 1e9             # 샘플링 속도 (1 GHz)
    probe_center_freq: float = 45e6        # 탐촉자 중심주파수 (45 MHz)
    bit_depth: int = 15                    # ADC Data Range (+- 2^15)
    
    # 3. 신호 처리 및 정렬 설정
    filter_lowcut_MHz: float = 0.0
    filter_highcut_MHz: float = 0.0
    filter_order: int = 2
    align_method: str = "envelope_peak" 
    align_pre_samples: int = 100    
    align_post_samples: int = 500   
    
    # 4. TGC & Log Compression 설정
    tgc_enable: bool = True
    tgc_start_sample_from_align: int = 0
    tgc_slope_dB: float = 0.02 
    log_cmp_alpha: float = 0.003 
    log_cmp_dynamic_range_dB: float = 35.0 
    
    # 5. 외부 문서 및 반전 탐지 설정
    ref_template_csv_path: str = "Document 26.09.01/ref.csv"
    number_of_samples_in_Whole_Abeam: int = 0
    phase_inv_search_start_offset_from_align: int = 100
    phase_inv_search_end_offset_from_align: int = 480
    phase_inv_neg_threshold: float = -0.4 
    phase_inv_roi_ratio: float = 1.0 
    phase_inv_whole_pos_ratio: float = 0.15 

    # 6. C-Scan 설정
    cscan_gate_start: int = 1
    cscan_gate_end: int = 180
    cscan_stretch_mode: str = 'absolute' 
    cscan_merge_mode: str = 'max' 

    def __post_init__(self): 
        self.update_filter_cutoffs()

    def update_filter_cutoffs(self) -> None:
        fc_mhz = self.probe_center_freq / 1e6
        self.filter_lowcut_MHz = max(0.0, fc_mhz / 3.0)
        self.filter_highcut_MHz = fc_mhz * 2.0


@dataclass
class CLayer:
    data_8bit_2d_map: np.ndarray
    depth_start: int
    depth_end: int
    gate_mode: str = 'max'
    slice_index: int = 0


# [개선] 순수 데이터 컨테이너 (slots=True 메모리 최적화 유지)
@dataclass(slots=True) 
class UltrasoundCubeData: 
    file_paths: List[str] = field(default_factory=list)

    num_rows: int = 0
    num_cols: int = 0
    num_samples: int = 0

    # 3D Cubes (볼륨 데이터)
    raw_3d_cube: Optional[np.ndarray] = None                  # int16
    filtered_3d_cube: Optional[np.ndarray] = None             # float32
    env_3d_cube: Optional[np.ndarray] = None                  # float32

    # B-Scan & C-Scan 재료 3D Cubes
    roi_3d_cube_8bit_for_B: Optional[np.ndarray] = None       # uint8
    roi_3d_cube_float_for_A: Optional[np.ndarray] = None      # float32
    roi_3d_env_cube_float_for_A: Optional[np.ndarray] = None  # float32

    # 3D FFT Magnitude Cubes
    raw_fft_mag_3d_cube: Optional[np.ndarray] = None          # float32
    filtered_fft_mag_3d_cube: Optional[np.ndarray] = None     # float32

    # 2D Maps (계산 결과 참조)
    align_map_envelope_peak: Optional[np.ndarray] = None       # int
    align_map_cross_corr: Optional[np.ndarray] = None          # int
    phase_inv_map: Optional[np.ndarray] = None                 # int

    # Shared Axes & Reference Signal
    shared_fft_freqs_MHz: Optional[np.ndarray] = None
    shared_sample_indices: Optional[np.ndarray] = None
    ref_template: Optional[np.ndarray] = None

    c_layers: List[CLayer] = field(default_factory=list)
    active_cscan_map: Optional[np.ndarray] = None


# ==========================================
# 2. Control Layer (Engine)
# ==========================================

class UltrasoundProcessorEngine:
    def __init__(self, data: UltrasoundCubeData, publisher_in: DataEventPublisher) -> None: 
        self.data: UltrasoundCubeData = data
        self.state: AppState = AppState()  # 통합 상태 관리 객체 소유
        self.publisher: DataEventPublisher = publisher_in

    def load_files_to_cube(self, file_paths: List[str]) -> bool: 
        if not file_paths:
            return False

        try:        
            self.data.file_paths = file_paths
            self.data.num_rows = len(file_paths)
            df_first = pd.read_csv(file_paths[0], header=None)
            self.data.num_cols = len(df_first)
            self.data.num_samples = len(df_first.iloc[0].dropna().values)
            
            self.data.raw_3d_cube = np.zeros((self.data.num_rows, self.data.num_cols, self.data.num_samples), dtype=np.int16)

            for row_idx, fpath in enumerate(file_paths): 
                df = pd.read_csv(fpath, header=None).dropna(axis=1, how='all')
                self.data.raw_3d_cube[row_idx] = df.values.astype(np.int16)

            self.data.shared_sample_indices = np.arange(self.data.num_samples, dtype=int)
            time_dist = 1.0 / self.state.sampling_rate
            freq_hz = np.fft.rfftfreq(self.data.num_samples, d=time_dist)
            self.data.shared_fft_freqs_MHz = freq_hz / 1e6
            
            self._load_ref_template()
            self.process_cube_pipeline()

            # 선택 상태 초기화
            self.state.active_row = 0
            self.state.active_col = 0

            self.publisher.notify("DATA_LOADED")
            return True

        except Exception as e:
            print(f"csv파일들->cube 과정에서 문제가 발생 : {e}")
            return False

    def _load_ref_template(self) -> None:
        _path = self.state.ref_template_csv_path
        if os.path.exists(_path):
            try:
                df = pd.read_csv(_path, header=None, nrows=1)
                self.data.ref_template = df.iloc[0].dropna().values.astype(np.float32)
            except Exception as e:
                print(f"Ref template 로드 오류: {e}")
                self.data.ref_template = None

    def process_cube_pipeline(self) -> None:
        if self.data.raw_3d_cube is None:
            raise ValueError("Raw cube가 할당되지 않았습니다.")

        self.data.filtered_3d_cube = self.apply_bandpass_filter(self.data.raw_3d_cube)
        self.data.env_3d_cube = self.extract_envelope(self.data.filtered_3d_cube)

        self.data.raw_fft_mag_3d_cube = self.compute_fft_cube(self.data.raw_3d_cube, data_samples=self.data.num_samples)
        self.data.filtered_fft_mag_3d_cube = self.compute_fft_cube(self.data.filtered_3d_cube, data_samples=self.data.num_samples)
        
        self.compute_align_maps()
        self.update_roi_cube_A_B_C()

    def set_active_selection(self, row_in: int, col_in: int) -> None: 
        row_changed = (self.state.active_row != row_in)
        col_changed = (self.state.active_col != col_in)

        self.state.active_row = max(0, min(row_in, self.data.num_rows - 1))
        self.state.active_col = max(0, min(col_in, self.data.num_cols - 1))

        if row_changed:
            self.publisher.notify("ROW_CHANGED") 

        if row_changed or col_changed:
            self.publisher.notify("SELECTION_CHANGED") 

    def set_cscan_parameters(self, d_start: int, d_end: int, stretch_mode: str, merge_mode: str) -> None:
        self.state.cscan_gate_start = d_start
        self.state.cscan_gate_end = d_end
        self.state.cscan_stretch_mode = stretch_mode
        self.state.cscan_merge_mode = merge_mode

        if self.data.roi_3d_cube_8bit_for_B is not None:
            self.generate_single_cscan(d_start, d_end)

    def generate_single_cscan(self, d_start: int, d_end: int) -> None: 
        layer = self.extract_cscan_layer(d_start, d_end, gate_mode=self.state.cscan_merge_mode)
        stretched_map = self.apply_histogram_stretch(layer.data_8bit_2d_map, mode=self.state.cscan_stretch_mode)
        self.data.active_cscan_map = stretched_map
        self.publisher.notify("CSCAN_UPDATED")

    def get_bscan_2d_map(self, row_idx: int) -> np.ndarray:
        if self.data.roi_3d_cube_8bit_for_B is None:
            raise ValueError("B-Scan Data Cube가 준비되지 않았습니다.")
        r = max(0, min(row_idx, self.data.num_rows - 1))
        return self.data.roi_3d_cube_8bit_for_B[r].T

    def extract_cscan_layer(self, depth_start: int, depth_end: int, gate_mode: str = 'max') -> CLayer:
        if self.data.roi_3d_cube_8bit_for_B is None:
            raise ValueError("B-Scan Data Cube(uint8)가 준비되지 않았습니다")

        s_idx = max(0, depth_start + self.state.align_pre_samples)
        e_idx = min(self.data.roi_3d_cube_8bit_for_B.shape[-1], depth_end + self.state.align_pre_samples)

        gate_data = self.data.roi_3d_cube_8bit_for_B[:, :, s_idx:e_idx]

        if gate_data.shape[-1] == 0:
            cscan_2d = np.zeros((self.data.num_rows, self.data.num_cols), dtype=np.uint8)
        elif gate_mode == "max":
            cscan_2d = np.max(gate_data, axis=-1).astype(np.uint8)
        elif gate_mode == "mean":
            cscan_2d = np.mean(gate_data, axis=-1).astype(np.uint8)
        else: 
            cscan_2d = np.mean(gate_data, axis=-1).astype(np.uint8)

        return CLayer(data_8bit_2d_map=cscan_2d, depth_start=depth_start, depth_end=depth_end, gate_mode=gate_mode)

    def apply_histogram_stretch(self, cscan_2d: np.ndarray, mode: str = 'absolute', abs_range: Tuple[int, int] = (10, 245), n_std: float = 2.0) -> np.ndarray:
        img_float = cscan_2d.astype(np.float32)

        if mode == 'absolute':
            min_val, max_val = abs_range
        elif mode == 'relative_std':
            mean_v = np.mean(img_float)
            std_v = np.std(img_float)
            min_val = max(0.0, mean_v - n_std * std_v)
            max_val = min(255.0, mean_v + n_std * std_v)
        else:
            min_val, max_val = abs_range

        if max_val <= min_val: 
            max_val = min_val + 1.0

        stretched = (img_float - min_val) / (max_val - min_val) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)

    def set_align_method(self, method_in: str) -> None:
        self.state.align_method = method_in
        self.update_roi_cube_A_B_C()
        self.publisher.notify("ROI_UPDATED")

    def update_active_align_map_pointer(self) -> None:
        if self.state.align_method == 'cross_corr' and self.data.align_map_cross_corr is not None:
            self.state.active_align_map = self.data.align_map_cross_corr
        else:
            self.state.active_align_map = self.data.align_map_envelope_peak

    def apply_bandpass_filter(self, signal: np.ndarray, axis_in: int = -1) -> np.ndarray:
        nyquist: float = 0.5 * self.state.sampling_rate
        low: float = max(0.001, min((self.state.filter_lowcut_MHz * 1e6) / nyquist, 0.98))
        high: float = max(0.002, min((self.state.filter_highcut_MHz * 1e6) / nyquist, 0.99))

        if low >= high:
            high = min(low + 0.01, 0.99)

        b, a = butter(self.state.filter_order, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=axis_in).astype(np.float32)

    def extract_envelope(self, signal_array: np.ndarray, axis_in: int = -1) -> np.ndarray:
        return np.abs(hilbert(signal_array, axis=axis_in)).astype(np.float32)

    def compute_fft_cube(self, signal_array: np.ndarray, data_samples: int, axis_in: int = -1) -> np.ndarray:
        return np.abs(np.fft.rfft(signal_array, axis=axis_in) / data_samples).astype(np.float32)

    def compute_align_maps(self) -> None:
        self.data.align_map_envelope_peak = np.argmax(self.data.env_3d_cube, axis=-1).astype(int)

        self.data.align_map_cross_corr = np.zeros((self.data.num_rows, self.data.num_cols), dtype=int)
        self.data.phase_inv_map = np.full((self.data.num_rows, self.data.num_cols), -1, dtype=int)

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

                    s_idx = pos_idx + inv_start
                    e_idx = min(pos_idx + inv_end, len(whole_sig))

                    if s_idx < e_idx: 
                        roi_corr = corr[s_idx:e_idx]
                        min_rel_idx = np.argmin(roi_corr) 
                        neg_val = roi_corr[min_rel_idx]
                        pos_val_in_whole_sig = corr[pos_idx]
                        max_val_roi = np.max(roi_corr) 

                        cond1 = neg_val < th_neg
                        cond2 = abs(neg_val) > (max_val_roi * th_roi_ratio)
                        cond3 = abs(neg_val) > (pos_val_in_whole_sig * th_pos_ratio)

                        if cond1 and cond2 and cond3: 
                            self.data.phase_inv_map[r, c] = s_idx + min_rel_idx
        else:
            self.data.align_map_cross_corr = self.data.align_map_envelope_peak.copy()

    def apply_tgc(self, roi_signal_3d_cube: np.ndarray) -> np.ndarray: 
        if not self.state.tgc_enable: 
            return roi_signal_3d_cube

        roi_len = roi_signal_3d_cube.shape[-1]
        indices = np.arange(roi_len)
        depth_offset = np.maximum(0, indices - self.state.tgc_start_sample_from_align)
        gain_dB = depth_offset * self.state.tgc_slope_dB
        tgc_gain = (10.0 ** (gain_dB / 20.0)).astype(np.float32)
        return roi_signal_3d_cube * tgc_gain

    def convert_to_8bit_log(self, roi_signal_3d_cube: np.ndarray) -> np.ndarray:
        data_safe = np.maximum(roi_signal_3d_cube, 0.0)
        alpha = self.state.log_cmp_alpha
        dr_dB = self.state.log_cmp_dynamic_range_dB
        data_log = 20.0 * np.log10(1.0 + alpha * data_safe)
        max_val = np.max(data_log)
        min_cutoff = max_val - dr_dB
        data_log = np.clip(data_log, min_cutoff, max_val)
 
        norm_data = (data_log - min_cutoff) / dr_dB
        _data_8bit = (norm_data * 255.0).astype(np.uint8)
        return _data_8bit

    def update_roi_cube_A_B_C(self) -> None:
        pre: int = self.state.align_pre_samples
        post: int = self.state.align_post_samples
        roi_len: int = pre + post

        roi_3d_signal_cube: np.ndarray = np.zeros((self.data.num_rows, self.data.num_cols, roi_len), dtype=np.float32)
        self.update_active_align_map_pointer()
        
        active_align_map: Optional[np.ndarray] = self.state.active_align_map

        for r in range(self.data.num_rows):
            for c in range(self.data.num_cols):
                a_idx = active_align_map[r, c]
                r_start = a_idx - pre
                r_end = a_idx + post
                s_start = max(0, r_start)
                s_end = min(self.data.num_samples, r_end)
                t_start = s_start - r_start 
                t_end = t_start + (s_end - s_start) 
            
                if s_start < s_end:
                    roi_3d_signal_cube[r, c, t_start:t_end] = self.data.filtered_3d_cube[r, c, s_start:s_end]

        self.data.roi_3d_cube_float_for_A = self.apply_tgc(roi_3d_signal_cube) 
        self.data.roi_3d_env_cube_float_for_A = self.extract_envelope(self.data.roi_3d_cube_float_for_A)
        self.data.roi_3d_cube_8bit_for_B = self.convert_to_8bit_log(self.data.roi_3d_env_cube_float_for_A)
        self.generate_single_cscan(self.state.cscan_gate_start, self.state.cscan_gate_end)


# ==========================================
# 3. View Layer (Tkinter Interface)
# ==========================================

class UltrasoundSignalViewer:
    def __init__(self): 
        self.window = tk.Tk()
        self.window.title("Ultrasound Signal Viewer (Linear Pub-Sub)")
        self.window.geometry("1280x800")

        # 인프라 및 핵심 모델 생성
        self.publisher: DataEventPublisher = DataEventPublisher()
        self.data: UltrasoundCubeData = UltrasoundCubeData()
        self.engine: UltrasoundProcessorEngine = UltrasoundProcessorEngine(data=self.data, publisher_in=self.publisher)

        self.view_mode_var = tk.StringVar(value='raw')
        self.align_method_var = tk.StringVar(value=self.state.align_method)

        self.bscan_img_display: Optional[matplotlib.image.AxesImage] = None 
        self.line_bscan_cursor: Optional[Line2D] = None 	
        self.last_ascan_update_time = 0.0					

        self.phase_inverse_var = tk.StringVar(value='no_apply') 	
        self.bscan_2d_gray: Optional[np.ndarray] = None 		
        self._bscan_rgb_cache: Optional[np.ndarray] = None 

        self.cscan_merge_var = tk.StringVar(value=self.state.cscan_merge_mode)
        self.cscan_stretch_var = tk.StringVar(value=self.state.cscan_stretch_mode)

        self.cscan_img_display: Optional[matplotlib.image.AxesImage] = None
        self.line_cscan_horiz: Optional[Line2D] = None
        self.line_cscan_vert: Optional[Line2D] = None

        self.create_widgets()
        self.register_event_subscriptions()

    @property
    def state(self) -> AppState:
        return self.engine.state

    def create_widgets(self): 
        control_frame = ttk.LabelFrame(self.window, text='Control Panel', padding=6)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        btn_open = ttk.Button(control_frame, text='Open CSVs', command=self.open_csvs)
        btn_open.grid(row=0, column=0, padx=(0, 5), pady=2)

        self.lbl_shape_info = ttk.Label(control_frame, text='0 Rows * 0 Cols', font=("Arial", 12, "bold"), foreground="#2e7d32")
        self.lbl_shape_info.grid(row=0, column=1, padx=10, pady=2, sticky="w")

        self.lbl_loaded_info = ttk.Label(control_frame, text="No files loaded", font=("Arial", 10), foreground="#4caf50")
        self.lbl_loaded_info.grid(row=1, column=1, padx=10, pady=2, sticky="w")

        ttk.Separator(control_frame, orient='vertical').grid(row=0, column=2, rowspan=2, sticky="ns", padx=10)

        # Spinboxes
        ttk.Label(control_frame, text='Row Index ', font=("Segoe UI", 11, "bold")).grid(row=0, column=3, padx=2, pady=2)
        self.spin_row = ttk.Spinbox(control_frame, from_=0, to=0, width=5, command=self.on_row_change)
        self.spin_row.grid(row=0, column=4, padx=5, pady=2)
        self.spin_row.bind('<Return>', lambda e: self.on_row_change())

        ttk.Label(control_frame, text='Col Index').grid(row=1, column=3, padx=2, pady=2, sticky="e")
        self.spin_col = ttk.Spinbox(control_frame, from_=0, to=0, width=5, command=self.on_col_change)
        self.spin_col.grid(row=1, column=4, padx=5, pady=2)
        self.spin_col.bind('<Return>', lambda e: self.on_col_change())

        ttk.Separator(control_frame, orient='vertical').grid(row=0, column=5, rowspan=2, sticky="ns", padx=10)

        # BandPass Radio
        ttk.Label(control_frame, text='BandPass', font=("Arial", 10, "bold")).grid(row=0, column=6, rowspan=2, padx=5)
        low_str = f"{self.state.filter_lowcut_MHz:.1f}"
        high_str = f"{self.state.filter_highcut_MHz:.1f}"
        ttk.Radiobutton(control_frame, text=f'Filtered {low_str}-{high_str}MHz', variable=self.view_mode_var, value='filtered', command=self.update_ascan_plots).grid(row=0, column=7, padx=5, pady=2, sticky="w")
        ttk.Radiobutton(control_frame, text='Raw Data', variable=self.view_mode_var, value='raw', command=self.update_ascan_plots).grid(row=1, column=7, padx=5, pady=2, sticky="w")

        ttk.Separator(control_frame, orient='vertical').grid(row=0, column=8, rowspan=2, sticky="ns", padx=10)
        
        # Align method Frame
        ttk.Label(control_frame, text='Align Method:', font=("Segoe UI", 9, "bold")).grid(row=0, column=9, padx=(0, 5), pady=2)
        ttk.Radiobutton(control_frame, text='Envelope Peak', variable=self.align_method_var, value='envelope_peak', command=self.on_align_change).grid(row=0, column=10, padx=3, pady=2, sticky='w')
        ttk.Radiobutton(control_frame, text='Cross corr', variable=self.align_method_var, value='cross_corr', command=self.on_align_change).grid(row=1, column=10, padx=3, pady=2, sticky='w')

        ttk.Separator(control_frame, orient="vertical").grid(row=0, column=11, rowspan=2, sticky="ns", padx=10)

        # Phase Inverse Control Group
        ttk.Label(control_frame, text='Phase Inverse :', font=("Segoe UI", 9, "bold")).grid(row=0, column=12, padx=(0, 5), pady=2)
        ttk.Radiobutton(control_frame, text="Blue Apply", variable=self.phase_inverse_var, value="blue_apply", command=self.on_phase_inv_toggle).grid(row=0, column=13, padx=3, pady=2, sticky='w')
        ttk.Radiobutton(control_frame, text="No Apply", variable=self.phase_inverse_var, value="no_apply", command=self.on_phase_inv_toggle).grid(row=1, column=13, padx=3, pady=2, sticky='w')

        ttk.Separator(control_frame, orient="vertical").grid(row=0, column=13, rowspan=2, sticky="ns", padx=8)

        # C-Scan Depth Settings
        ttk.Label(control_frame, text='Depth start', font=("Segoe UI", 9, "bold")).grid(row=0, column=14, padx=2, pady=2, sticky="e")
        self.spin_depth_start = ttk.Spinbox(control_frame, from_=-100, to=500, width=5, command=self.on_cscan_setting_change)
        self.spin_depth_start.grid(row=0, column=15, padx=3, pady=2)
        self.spin_depth_start.delete(0, tk.END); self.spin_depth_start.insert(0, str(self.state.cscan_gate_start))
        self.spin_depth_start.bind('<Return>', lambda e: self.on_cscan_setting_change())

        ttk.Label(control_frame, text='Depth end', font=("Segoe UI", 9, "bold")).grid(row=1, column=14, padx=2, pady=2, sticky="e")
        self.spin_depth_end = ttk.Spinbox(control_frame, from_=-100, to=500, width=5, command=self.on_cscan_setting_change)
        self.spin_depth_end.grid(row=1, column=15, padx=3, pady=2)
        self.spin_depth_end.delete(0, tk.END); self.spin_depth_end.insert(0, str(self.state.cscan_gate_end))
        self.spin_depth_end.bind('<Return>', lambda e: self.on_cscan_setting_change())

        ttk.Separator(control_frame, orient="vertical").grid(row=0, column=16, rowspan=2, sticky="ns", padx=8)

        ttk.Label(control_frame, text='Merge', font=("Segoe UI", 9, "bold")).grid(row=0, column=17, rowspan=2, padx=2)
        ttk.Radiobutton(control_frame, text='Max', variable=self.cscan_merge_var, value='max', command=self.on_cscan_setting_change).grid(row=0, column=18, padx=2, pady=2, sticky='w')
        ttk.Radiobutton(control_frame, text='Mean', variable=self.cscan_merge_var, value='mean', command=self.on_cscan_setting_change).grid(row=1, column=18, padx=2, pady=2, sticky='w')

        ttk.Separator(control_frame, orient="vertical").grid(row=0, column=19, rowspan=2, sticky="ns", padx=8)

        ttk.Label(control_frame, text='Stretch', font=("Segoe UI", 9, "bold")).grid(row=0, column=20, rowspan=2, padx=2)
        ttk.Radiobutton(control_frame, text='Abs', variable=self.cscan_stretch_var, value='absolute', command=self.on_cscan_setting_change).grid(row=0, column=21, padx=2, pady=2, sticky='w')
        ttk.Radiobutton(control_frame, text='Relative', variable=self.cscan_stretch_var, value='relative_std', command=self.on_cscan_setting_change).grid(row=1, column=21, padx=2, pady=2, sticky='w')

        # Main Plot Layout
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = matplotlib.figure.Figure(figsize=(13, 7), dpi=100)
        gs = GridSpec(3, 2, figure=self.fig, width_ratios=[1, 1.2])

        self.ax_roi = self.fig.add_subplot(gs[0, 0])
        self.ax_whole = self.fig.add_subplot(gs[1, 0])
        self.ax_fft = self.fig.add_subplot(gs[2, 0])

        self.ax_bscan = self.fig.add_subplot(gs[0:1, 1])
        self.ax_cscan = self.fig.add_subplot(gs[1:3, 1])

        self.setup_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('motion_notify_event', self.on_bscan_hover)
        self.canvas.mpl_connect('button_press_event', self.on_cscan_click)
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def setup_plots(self): 
        self.ax_roi.set_title("ROI Align Signal", fontsize=9, fontweight='bold')
        self.ax_roi.grid(True, linestyle='--', alpha=0.5)
        self.line_roi_sig_for_A = self.ax_roi.plot([], [], color='#1f77b4', lw=1.0, label='Signal')[0]
        self.line_roi_env = self.ax_roi.plot([], [], color='#ff7f0e', lw=1.0, ls='--', label='Envelope')[0]
        self.ax_roi.legend(loc='upper right', fontsize=7)
        self.ax_roi.set_ylim(-32768, 32768)
	
        self.ax_whole.set_title('Raw Ascan', fontsize=9, fontweight='bold')
        self.ax_whole.grid(True, linestyle='--', alpha=0.5)
        self.line_whole_sig = self.ax_whole.plot([], [], color='#1f77b4', lw=0.8)[0]
        self.line_whole_env = self.ax_whole.plot([], [], color='#ff7f0e', lw=1.0, ls='--')[0]
        self.line_align_mark = self.ax_whole.axvline(x=0, color='red', ls=':', lw=1)
        self.line_inv_mark = self.ax_whole.axvline(x=0, color='purple', ls=':', lw=1, visible=False)
        self.ax_whole.set_ylim(-32768, 32768)
	
        self.ax_fft.set_title("Raw FFT : Peak", fontsize=9, fontweight='bold')
        self.ax_fft.grid(True, linestyle='--', alpha=0.5)
        self.line_fft = self.ax_fft.plot([], [], color='#d62728', lw=1.0)[0]
        self.line_fft_peak = self.ax_fft.axvline(x=0, color='green', ls='--', lw=1)
	
        self.ax_bscan.set_title("B SCAN", fontsize=11, fontweight='bold')
        self.ax_bscan.set_ylabel("Depth Samples")
        self.line_bscan_cursor = self.ax_bscan.axvline(x=0, color='#00ff00', lw=1.5, visible=False)

        self.ax_cscan.set_title("C-SCAN", fontsize=11, fontweight='bold')
        self.line_cscan_horiz = self.ax_cscan.axhline(y=0, color='#00ff00', lw=1.0, visible=False)
        self.line_cscan_vert = self.ax_cscan.axvline(x=0, color='#00ff00', lw=1.0, visible=False)

        self.fig.tight_layout()

    def register_event_subscriptions(self):
        self.publisher.subscribe("DATA_LOADED", self.on_event_datas_loaded)
        self.publisher.subscribe("ROW_CHANGED", self.update_loaded_label)
        self.publisher.subscribe("ROW_CHANGED", self.render_bscan)
        self.publisher.subscribe("SELECTION_CHANGED", self.update_ascan_plots)
        self.publisher.subscribe("ROI_UPDATED", self.on_event_roi_updated)
        self.publisher.subscribe("CSCAN_UPDATED", self.render_cscan)
        self.publisher.subscribe("CSCAN_SETTINGS_CHANGED", self.on_event_cscan_settings_changed)

    # Subscriber Callbacks
    def on_event_datas_loaded(self):
        self.spin_row.config(from_=0, to=self.data.num_rows - 1)
        self.spin_col.config(from_=0, to=self.data.num_cols - 1)

        self.spin_row.delete(0, tk.END); self.spin_row.insert(0, "0")
        self.spin_col.delete(0, tk.END); self.spin_col.insert(0, "0")

        self.lbl_shape_info.config(text=f'{self.data.num_rows}Rows*{self.data.num_cols}Cols')
        self.update_loaded_label()

        self.ax_whole.set_xlim(0, self.data.num_samples)
        self.line_whole_sig.set_xdata(self.data.shared_sample_indices)
        self.line_whole_env.set_xdata(self.data.shared_sample_indices)

        self.line_fft.set_xdata(self.data.shared_fft_freqs_MHz)
        self.ax_fft.set_xlim(self.state.filter_lowcut_MHz, self.state.filter_highcut_MHz)
				
        roi_x = np.arange(-self.state.align_pre_samples, self.state.align_post_samples)
        self.ax_roi.set_xlim(-self.state.align_pre_samples, self.state.align_post_samples)
        self.line_roi_sig_for_A.set_xdata(roi_x)
        self.line_roi_env.set_xdata(roi_x)

        self.render_bscan()
        self.render_cscan()
        self.update_ascan_plots()

    def on_event_roi_updated(self):
        self.render_bscan()
        self.render_cscan()
        self.update_ascan_plots()

    def on_event_cscan_settings_changed(self) -> None:
        self.render_cscan()

    # UI Handlers
    def on_cscan_setting_change(self) -> None:
        try:
            d_start = int(self.spin_depth_start.get())
            d_end = int(self.spin_depth_end.get())
        except ValueError:
            d_start = self.state.cscan_gate_start
            d_end = self.state.cscan_gate_end

        merge_mode = self.cscan_merge_var.get()
        stretch_mode = self.cscan_stretch_var.get()

        self.engine.set_cscan_parameters(d_start, d_end, stretch_mode, merge_mode)
        self.publisher.notify("CSCAN_SETTINGS_CHANGED")

    def on_row_change(self) -> None: 
        try:
            r = int(self.spin_row.get())
            c = int(self.spin_col.get())
            self.engine.set_active_selection(r, c)
        except: 
            r = self.state.active_row
            c = self.state.active_col
            self.engine.set_active_selection(r, c)

    def on_col_change(self) -> None:
        try:
            r = int(self.spin_row.get())
            c = int(self.spin_col.get())
            self.engine.set_active_selection(r, c) 
        except:
            r = self.state.active_row
            c = self.state.active_col
            self.engine.set_active_selection(r, c)

    def on_align_change(self): 
        self.engine.set_align_method(self.align_method_var.get())

    def on_phase_inv_toggle(self):
        self.render_bscan()
        self.update_ascan_plots()
	
    def open_csvs(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if paths:
            self.engine.load_files_to_cube(list(paths))

    # Renderers
    def render_bscan(self): 
        if self.data.roi_3d_cube_8bit_for_B is None:
            return

        r = self.state.active_row
        self.bscan_2d_gray = self.data.roi_3d_cube_8bit_for_B[r].T 
        h, w = self.bscan_2d_gray.shape

        if (self._bscan_rgb_cache is None) or (self._bscan_rgb_cache.shape != (h, w, 3)):
            self._bscan_rgb_cache = np.empty((h, w, 3), dtype=np.uint8)

        self._bscan_rgb_cache[:, :, 0] = self.bscan_2d_gray
        self._bscan_rgb_cache[:, :, 1] = self.bscan_2d_gray
        self._bscan_rgb_cache[:, :, 2] = self.bscan_2d_gray
	
        pre = self.state.align_pre_samples
        post = self.state.align_post_samples
        extent = [0, w-1, post, -pre]

        cond1 = self.phase_inverse_var.get() == "blue_apply"
        cond2 = self.data.phase_inv_map is not None
        is_blue_apply = cond1 and cond2
	
        if is_blue_apply:
            active_align_map = self.state.active_align_map
            for c in range(w): 
                inv_abs_idx = self.data.phase_inv_map[r, c]
                if inv_abs_idx != -1:
                    align_abs_idx = active_align_map[r, c]
                    rel_idx = inv_abs_idx - align_abs_idx + pre 
                    if 0 <= rel_idx < h: 
                        self._bscan_rgb_cache[rel_idx, c] = [0, 0, 255]

        if self.bscan_img_display is not None:
            self.bscan_img_display.set_data(self._bscan_rgb_cache) 
            self.bscan_img_display.set_extent(extent)
        else: 
            self.bscan_img_display = self.ax_bscan.imshow( 
                self._bscan_rgb_cache, aspect='auto', origin='upper', extent=extent
            )
            self.line_bscan_cursor.set_visible(True)

    def render_cscan(self):
        if self.data.active_cscan_map is None:
            return

        if self.cscan_img_display is not None:
            self.cscan_img_display.set_data(self.data.active_cscan_map)
        else:
            self.cscan_img_display = self.ax_cscan.imshow(
                self.data.active_cscan_map, cmap='gray', aspect='auto', origin='upper'
            )
            self.line_cscan_horiz.set_visible(True)
            self.line_cscan_vert.set_visible(True)

        self.canvas.draw_idle()

    def update_ascan_plots(self) -> None:  
        if self.data.raw_3d_cube is None: 
            return
		
        r, c = self.state.active_row, self.state.active_col
        mode = self.view_mode_var.get()

        if str(r) != self.spin_row.get():
            self.spin_row.delete(0, tk.END); self.spin_row.insert(0, str(r))
        if str(c) != self.spin_col.get():
            self.spin_col.delete(0, tk.END); self.spin_col.insert(0, str(c))
	
        if mode == 'raw': 
            sig = self.data.raw_3d_cube[r, c]
            env = self.data.env_3d_cube[r, c]
            fft_mag = self.data.raw_fft_mag_3d_cube[r, c] 
        else:
            sig = self.data.filtered_3d_cube[r, c]
            env = self.data.env_3d_cube[r, c]
            fft_mag = self.data.filtered_fft_mag_3d_cube[r, c] 
	
        self.line_whole_sig.set_ydata(sig)
        self.line_whole_env.set_ydata(env)
	
        align_idx: int = self.state.active_align_map[r, c]
        self.line_align_mark.set_xdata([align_idx, align_idx])
        inv_idx = self.data.phase_inv_map[r, c]
	
        if inv_idx != -1:
            self.line_inv_mark.set_xdata([inv_idx, inv_idx])
            self.line_inv_mark.set_visible(True)
            self.ax_roi.set_title(f"ROI Align : {align_idx}, Inverse : {inv_idx}", fontsize=9, fontweight='bold')
        else: 
            self.line_inv_mark.set_visible(False)
            self.ax_roi.set_title(f"ROI Align : {align_idx}", fontsize=9, fontweight='bold')
            
        self.line_fft.set_ydata(fft_mag)
        peak_freq_idx = np.argmax(fft_mag)
        peak_freq_MHz = self.data.shared_fft_freqs_MHz[peak_freq_idx]
        self.ax_fft.set_title(f"FFT: Peak={peak_freq_MHz:.1f}MHz", fontsize=9, fontweight='bold')
        self.line_fft_peak.set_xdata([peak_freq_MHz, peak_freq_MHz])
	
        max_mag = np.max(fft_mag)
        self.ax_fft.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
	
        if self.data.roi_3d_cube_float_for_A is not None: 	
            self.line_roi_sig_for_A.set_ydata(self.data.roi_3d_cube_float_for_A[r, c])

        if self.data.roi_3d_env_cube_float_for_A is not None:
            self.line_roi_env.set_ydata(self.data.roi_3d_env_cube_float_for_A[r, c])
		
        self.line_bscan_cursor.set_xdata([c, c])
        self.line_cscan_horiz.set_ydata([r, r])
        self.line_cscan_vert.set_xdata([c, c])

        self.canvas.draw_idle()

    def update_loaded_label(self): 
        if not self.data.file_paths or self.state.active_row >= len(self.data.file_paths): 
            return
        filename = os.path.basename(self.data.file_paths[self.state.active_row])
        self.lbl_loaded_info.config(text=f"{filename} Loaded", foreground="green")

    # Mouse Events
    def on_cscan_click(self, event_in):
        if event_in.inaxes == self.ax_cscan and event_in.xdata is not None and event_in.ydata is not None:
            _col = int(round(event_in.xdata))
            _row = int(round(event_in.ydata))
            self.engine.set_active_selection(_row, _col)
						
    def on_bscan_hover(self, event): 
        if event.inaxes == self.ax_bscan and event.key == 'control':
            if event.xdata is not None:
                _col = int(round(event.xdata))
                if 0 <= _col < self.data.num_cols:
                    current_time = time.time()
                    if current_time - self.last_ascan_update_time > 0.1:
                        self.last_ascan_update_time = current_time
                        self.spin_col.delete(0, tk.END); self.spin_col.insert(0, str(_col)) 
                        self.engine.set_active_selection(self.state.active_row, _col)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = UltrasoundSignalViewer()
    app.run()