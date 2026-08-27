import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

def hex_to_signed16(hex_str):
    """4자리 16진수 문자열을 Signed 16비트 정수로 변환"""
    val = int(hex_str.strip(), 16)
    if val >= 0x8000:
        val -= 0x10000
    return val

class CsvConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("16진수-10진수 CSV 변환기")
        self.root.geometry("500x200")
        self.root.resizable(False, False)

        self.file_path = ""

        # 1. 파일 선택 영역
        self.btn_open = tk.Button(root, text="파일 Open", command=self.open_file, width=12, height=1)
        self.btn_open.pack(pady=(20, 5))

        # 2. 로드된 파일 경로 표시 Label
        self.lbl_file = tk.Label(root, text="선택된 파일 없음", fg="gray", wraplength=450)
        self.lbl_file.pack(pady=5)

        # 3. 변환 실행 버튼
        self.btn_convert = tk.Button(root, text="10진수 변환 실행", command=self.convert_file, width=15, height=2, bg="#4CAF50", fg="white")
        self.btn_convert.pack(pady=15)

    def open_file(self):
        # 파일 탐색기 창 열기
        path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.file_path = path
            self.lbl_file.config(text=f"선택된 파일: {os.path.basename(path)}", fg="black")

    def convert_file(self):
        if not self.file_path:
            messagebox.showwarning("경고", "먼저 변환할 CSV 파일을 선택하세요.")
            return

        # 저장할 파일 경로 자동 생성 (파일명_decimal.csv)
        dir_name, file_name = os.path.split(self.file_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{name}_decimal{ext}")

        try:
            with open(self.file_path, "r", encoding="utf-8") as infile, \
                 open(output_path, "w", newline="", encoding="utf-8") as outfile:
                
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                for row in reader:
                    converted_row = [hex_to_signed16(val) for val in row if val.strip()]
                    writer.writerow(converted_row)

            messagebox.showinfo("완료", f"변환이 완료되었습니다!\n\n저장 위치:\n{output_path}")
        except Exception as e:
            messagebox.showerror("오류", f"파일 처리 중 오류가 발생했습니다:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CsvConverterApp(root)
    root.mainloop()