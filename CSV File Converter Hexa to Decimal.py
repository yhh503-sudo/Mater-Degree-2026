import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


def hex_to_signed16(hex_str):
    """4자리 16진수 문자열을 Signed 16비트 정수로 변환"""
    val = int(hex_str.strip(), 16)
    if val >= 0x8000:
        val -= 0x10000
    return val


class CsvSliceConverterApp:

    def __init__(self, root):
        self.root = root
        self.root.title("16진수-10진수 CSV 컬럼 변환기")
        self.root.geometry("480x380")
        self.root.resizable(False, False)

        self.file_path = ""

        # 1. 파일 선택
        self.btn_open = tk.Button(
            root, text="파일 Open", command=self.open_file, width=15
        )
        self.btn_open.pack(pady=(15, 5))

        self.lbl_file = tk.Label(
            root, text="선택된 파일 없음", fg="gray", wraplength=440
        )
        self.lbl_file.pack(pady=5)

        # 2. 컬럼 설정 프레임
        frame_config = tk.Frame(root)
        frame_config.pack(pady=10)

        tk.Label(frame_config, text="시작 컬럼 Index:").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        self.entry_start = tk.Entry(frame_config, width=10, justify="center")
        self.entry_start.insert(0, "500")
        self.entry_start.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame_config, text="컬럼 개수:").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        self.entry_count = tk.Entry(frame_config, width=10, justify="center")
        self.entry_count.insert(0, "700")
        self.entry_count.grid(row=1, column=1, padx=5, pady=5)

        # 3. 진행 상황 (ProgressBar & Label)
        self.progress_bar = ttk.Progressbar(
            root, orient="horizontal", length=400, mode="determinate"
        )
        self.progress_bar.pack(pady=(10, 2))

        self.lbl_progress = tk.Label(root, text="진행률: 0.0% (0 / 0 Rows)")
        self.lbl_progress.pack(pady=5)

        # 4. 변환 실행 버튼
        self.btn_convert = tk.Button(
            root,
            text="슬라이싱 변환 실행",
            command=self.convert_file,
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white",
        )
        self.btn_convert.pack(pady=10)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.file_path = path
            self.lbl_file.config(
                text=f"선택된 파일: {os.path.basename(path)}", fg="black"
            )
            # 진행바 초기화
            self.progress_bar["value"] = 0
            self.lbl_progress.config(text="진행률: 0.0% (0 / 0 Rows)")

    def convert_file(self):
        if not self.file_path:
            messagebox.showwarning("경고", "먼저 CSV 파일을 선택하세요.")
            return

        try:
            start_idx = int(self.entry_start.get().strip())
            count = int(self.entry_count.get().strip())
            if start_idx < 0 or count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "입력 오류",
                "시작 Index는 0 이상의 정수, 개수는 1 이상의 정수여야 합니다.",
            )
            return

        end_idx = start_idx + count
        dir_name, file_name = os.path.split(self.file_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(
            dir_name, f"{name}_cols_{start_idx}_{end_idx}{ext}"
        )

        try:
            # 전체 라인 수 미리 계산 (ProgressBar 범위 설정용)
            with open(self.file_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f)

            if total_rows == 0:
                messagebox.showwarning("경고", "빈 파일입니다.")
                return

            self.progress_bar["maximum"] = total_rows

            # UI 갱신 간격 설정 (전체 행의 1% 마다 또는 최소 100행마다 갱신)
            update_interval = max(1, total_rows // 100)

            with open(
                self.file_path, "r", encoding="utf-8"
            ) as infile, open(
                output_path, "w", newline="", encoding="utf-8"
            ) as outfile:

                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                for current_row, row in enumerate(reader, start=1):
                    target_cols = row[start_idx:end_idx]
                    converted_row = [
                        hex_to_signed16(val)
                        for val in target_cols
                        if val.strip()
                    ]
                    writer.writerow(converted_row)

                    # 일정 간격(Batch)으로만 ProgressBar 업데이트하여 속도 저하 방지
                    if (
                        current_row % update_interval == 0
                        or current_row == total_rows
                    ):
                        percent = (current_row / total_rows) * 100
                        self.progress_bar["value"] = current_row
                        self.lbl_progress.config(
                            text=f"진행률: {percent:.1f}% ({current_row:,} / {total_rows:,} Rows)"
                        )
                        self.root.update_idletasks()  # GUI 강제 갱신

            messagebox.showinfo(
                "완료", f"변환 완료!\n\n저장 위치:\n{output_path}"
            )

        except Exception as e:
            messagebox.showerror(
                "오류", f"파일 처리 중 오류가 발생했습니다:\n{e}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = CsvSliceConverterApp(root)
    root.mainloop()