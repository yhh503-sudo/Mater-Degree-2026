import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image


def process_images():
    # 1. 파일 선택 창 열기 (여러 BMP 파일 선택)
    file_paths = filedialog.askopenfilenames(
        title="BMP 이미지 파일들을 선택하세요",
        filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")],
    )

    if not file_paths:
        return  # 파일 선택을 취소한 경우

    success_count = 0

    for path in file_paths:
        try:
            # 이미지 열기 및 넘파이 배열 변환
            img = Image.open(path)
            img_np = np.array(img)

            # --- 이미지 슬라이싱 조건 ---
            # 1) 열(Column) 처리: 항상 짝수 열(인덱스 1, 3, 5...) 제거 -> 홀수 열(인덱스 0, 2, 4...)만 선택
            #    Python 0-based index 기준: [:, ::2]
            img_col_filtered = img_np[:, ::2]

            # 2) 행(Row/줄) 처리:
            #    0-based index 기준:
            #    - 짝수 줄 (0, 2, 4번째 행 -> 1번째, 3번째, 5번째 줄) => _L.bmp
            #    - 홀수 줄 (1, 3, 5번째 행 -> 2번째, 4번째, 6번째 줄) => _R.bmp
            img_L = img_col_filtered[0::2]
            img_R = img_col_filtered[1::2]

            # 저장 경로 설정
            dir_name, full_name = os.path.split(path)
            base_name, _ = os.path.splitext(full_name)

            path_L = os.path.join(dir_name, f"{base_name}_L.bmp")
            path_R = os.path.join(dir_name, f"{base_name}_R.bmp")

            # 배열을 다시 이미지로 변환하여 저장
            Image.fromarray(img_L).save(path_L)
            Image.fromarray(img_R).save(path_R)

            success_count += 1

        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({path}): {e}")

    # 작업 완료 알림
    messagebox.showinfo(
        "완료",
        f"총 {len(file_paths)}개 중 {success_count}개 파일의 변환이 완료되었습니다!",
    )


# --- GUI 구성 (Tkinter) ---
def create_gui():
    root = tk.Tk()
    root.title("BMP 이미지 분할 프로그램")
    root.geometry("400x200")
    root.resizable(False, False)

    # 안내 레이블
    label_title = tk.Label(
        root, text="BMP 이미지 L/R 분할 프로그램", font=("Arial", 14, "bold")
    )
    label_title.pack(pady=20)

    label_desc = tk.Label(
        root,
        text="여러 BMP 파일을 선택하여 L / R 이미지를 추출합니다.\n(열: 짝수열 제거 / 행: 짝수줄=L, 홀수줄=R)",
        font=("Arial", 9),
        justify="center",
    )
    label_desc.pack(pady=5)

    # 파일 선택 및 변환 버튼
    btn_select = tk.Button(
        root,
        text="BMP 파일 선택 및 변환 시작",
        command=process_images,
        font=("Arial", 11),
        bg="#4CAF50",
        fg="white",
        padx=10,
        pady=5,
    )
    btn_select.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_gui()