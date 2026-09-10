import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 (CSV 파일명은 사용자에 맞게 수정)
filename = 'ultrasound_data.csv'  # 실제 파일명으로 변경하세요.
try:
    data = np.loadtxt(filename, delimiter=',')
except FileNotFoundError:
    print(f"Error: '{filename}' 파일을 찾을 수 없습니다. 파일명을 확인하세요.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# 데이터 형태 확인 및 처리
if data.ndim > 1:
    print("Warning: CSV 데이터가 1차원 배열이 아닙니다. 첫 번째 열(또는 행)만 사용합니다.")
    if data.shape[1] > 1: # 열이 여러개면 첫 번째 열 사용
        data = data[:, 0]
    else: # 행이 여러개고 열이 1개면 flattening
        data = data.flatten()


N = len(data)
if N == 0:
    print("Error: 데이터가 없습니다.")
    exit()

# 2. 파라미터 설정 및 FFT 계산
fs = 1.25e9  # 샘플링 주파수 (1.25 GHz)
t = np.arange(N) / fs # 타임 벡터 (시각화용)

# Hann 윈도우 적용 (스펙트럼 누설 감소)
window = np.hanning(N)
windowed_data = data * window

# FFT 계산
# 복소수 결과의 절대값을 취해 크기(magnitude) 스펙트럼 획득
fft_result = np.fft.fft(windowed_data)
# 주파수 축 생성 (양수 주파수만 사용)
freqs = np.fft.fftfreq(N, 1/fs)

# 양수 주파수 영역만 추출 (0 ~ fs/2)
positive_freqs_mask = freqs >= 0
freqs = freqs[positive_freqs_mask]
magnitude = np.abs(fft_result[positive_freqs_mask])

# 스펙트럼 정규화 (최대값 = 0dB)
magnitude_db = 20 * np.log10(magnitude / np.max(magnitude))

# 3. 주요 주파수 특성 계산
# 최대 크기를 가지는 주파수 (Peak Frequency, 중심 주파수 근사치)
peak_idx = np.argmax(magnitude_db)
center_freq = freqs[peak_idx]

# -6dB 대역폭 계산
minus_6db_indices = np.where(magnitude_db >= -6)[0]
if len(minus_6db_indices) >= 2:
    f_low = freqs[minus_6db_indices[0]]
    f_high = freqs[minus_6db_indices[-1]]
    bandwidth_6db = f_high - f_low
else:
    f_low = None
    f_high = None
    bandwidth_6db = None
    print("Warning: -6dB 대역폭을 계산할 수 없습니다 (신호 레벨이 너무 낮거나 대역폭이 너무 넓음).")


# 4. 이미지 생성 및 표시 (시각화)
plt.figure(figsize=(12, 8))

# 원본 시간 영역 파형 플롯 (선택 사항, 상단에 배치)
plt.subplot(2, 1, 1)
plt.plot(t * 1e6, data, label='Original Signal') # 시간 축을 us 단위로 변환
plt.title('Time Domain Ultrasound Waveform')
plt.xlabel('Time (us)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 주파수 영역 (FFT) 스펙트럼 플롯
plt.subplot(2, 1, 2)
# 주파수 축을 MHz 단위로 변환하여 플롯
plt.plot(freqs / 1e6, magnitude_db, label='FFT Spectrum (Normalized)')
plt.title('Frequency Domain Spectrum (FFT)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-80, 5]) # dB 축 범위 설정
plt.grid(True)

# 중심 주파수 표시
plt.axvline(x=center_freq / 1e6, color='r', linestyle='--', label=f'Center Freq: {center_freq/1e6:.2f} MHz')

# -6dB 레벨 선 및 대역폭 영역 표시
if f_low is not None and f_high is not None:
    plt.axhline(y=-6, color='g', linestyle=':', label='-6dB Level')
    plt.axvspan(f_low / 1e6, f_high / 1e6, color='g', alpha=0.3, label=f'-6dB BW: {bandwidth_6db/1e6:.2f} MHz')
    # 낮은/높은 -6dB 주파수 텍스트 표시
    plt.text(f_low / 1e6, -10, f'{f_low/1e6:.1f}', color='g', ha='right')
    plt.text(f_high / 1e6, -10, f'{f_high/1e6:.1f}', color='g', ha='left')

plt.legend(loc='upper right') # 범례 표시 위치 설정
plt.tight_layout() # 서브플롯 간격 조정

# 결과 이미지 저장 (또는 plt.show()로 화면에 표시)
plt.savefig('ultrasound_fft_result.png') # 이미지 파일로 저장
# plt.show() # 화면에 직접 표시하려면 이 줄의 주석을 해제하세요.

print(f"FFT 분석 결과가 'ultrasound_fft_result.png' 파일로 저장되었습니다.")
if bandwidth_6db is not None:
    print(f"계산된 중심 주파수: {center_freq/1e6:.2f} MHz")
    print(f"-6dB 대역폭: {bandwidth_6db/1e6:.2f} MHz ({f_low/1e6:.2f} MHz ~ {f_high/1e6:.2f} MHz)")