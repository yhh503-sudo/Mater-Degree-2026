import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. 데이터 준비
x = np.array([11.2, 15.8, 18.9, 19.5, 21.5, 21.8, 25.3, 26.4, 26.7, 29.1]).reshape(-1, 1)
y = np.array([21.5, 4.2, 1.8, 1.0, 1.0, 0.8, 3.8, 7.5, 4.3, 35.0])

# 2. 종속변수 log10 변환
y_log = np.log10(y)

# 3. 2차 다항 특성 생성 (x, x^2)
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# 4. 선형 회귀 모델 학습
model = LinearRegression()
model.fit(x_poly, y_log)

# 계수 출력
print(f"절편 (w0): {model.intercept_:.4f}")
print(f"계수 (w1, w2): {model.coef_[0]:.4f}, {model.coef_[1]:.5f}")

# 5. 예측 및 R-squared 계산
y_log_pred = model.predict(x_poly)
r2 = r2_score(y_log, y_log_pred)
print(f"R-squared: {r2 * 100:.1f}%")

# 6. 시각화 (이미지와 동일한 그래프 생성)
x_seq = np.linspace(11, 30, 100).reshape(-1, 1)
x_seq_poly = poly.transform(x_seq)
y_seq_log_pred = model.predict(x_seq_poly)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points') #이게 원본데이터로 보임
plt.plot(x_seq, 10**y_seq_log_pred, color='navy', label='Fitted Line (Exponential Scale)')
plt.yscale('log') # y축을 로그 스케일로 표시
plt.xlabel('MachineSetting')
plt.ylabel('Energy Consumption')
plt.title('Fitted Line Plot (Python)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()