import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from datetime import datetime

def process_images():
    # 탐색기를 열어 여러 장의 이미지 선택
    file_paths = filedialog.askopenfilenames(
        title="이미지 선택",
        filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")]
    )
    
    # 선택된 파일이 없으면 종료
    if not file_paths:
        return
        
    # 첫 번째 이미지를 기준으로 초기화
    # cv2는 이미지를 numpy 배열(BGR 형태)로 읽어옵니다.
    base_img = cv2.imread(file_paths[0])
    
    if base_img is None:
        messagebox.showerror("오류", f"이미지를 읽을 수 없습니다:\n{file_paths[0]}")
        return
        
    base_shape = base_img.shape
    max_img = base_img.copy()
    
    # 두 번째 이미지부터 순회하며 크기 비교 및 최댓값 갱신
    for path in file_paths[1:]:
        img = cv2.imread(path)
        
        if img is None:
            messagebox.showerror("오류", f"이미지를 읽을 수 없습니다:\n{path}")
            return
            
        # 이미지 사이즈(shape)가 하나라도 다르면 팝업창 띄우고 예외 처리
        if img.shape != base_shape:
            messagebox.showerror("오류", f"선택한 이미지들의 크기가 동일하지 않습니다.\n\n기준 크기: {base_shape}\n다른 크기: {img.shape}\n파일: {path}")
            return
            
        # 동일한 픽셀 좌표에 대해 두 이미지 중 더 큰 픽셀값으로 갱신
        max_img = np.maximum(max_img, img)

    # 현재 시간을 '시_분_초' 형태로 가져와 기본 파일명으로 설정
    current_time = datetime.now().strftime("%H_%M_%S")
    default_filename = f"{current_time}.bmp"

    # 합성된 이미지를 저장할 경로 지정
    save_path = filedialog.asksaveasfilename(
        title="결과 이미지 저장",
        initialfile=default_filename,
        defaultextension=".bmp",
        filetypes=[("BMP file", "*.bmp"), ("PNG file", "*.png"), ("JPEG file", "*.jpg")]
    )
    
    if save_path:
        cv2.imwrite(save_path, max_img)
        messagebox.showinfo("완료", "최댓값 합성 이미지가 성공적으로 저장되었습니다.")

# Tkinter UI 설정
root = tk.Tk()
root.title("Max Pixel Image Generator")
root.geometry("300x150")
root.eval('tk::PlaceWindow . center') # 창을 화면 중앙에 배치

# Open 버튼 생성
open_btn = tk.Button(root, text="Open", command=process_images, width=20, height=3)
open_btn.pack(expand=True)

# UI 실행
root.mainloop()