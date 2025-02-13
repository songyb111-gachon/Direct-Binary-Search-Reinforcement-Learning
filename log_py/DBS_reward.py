import tkinter as tk
from tkinter import filedialog, messagebox
import re
import numpy as np
from collections import defaultdict

# 로그 파일에서 데이터 추출
def parse_log_file(file_path, step_unit, max_step):
    step_data = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # 정규표현식으로 데이터 매칭
            step_matches = re.finditer(
                r"Step:\s+(\d+)\s+"  # Step
                r"PSNR Before:\s+[\d.]+\s+\|\s+PSNR After:\s+[\d.]+\s+\|\s+Change:\s+[\d.e+-]+\s+\|\s+Diff:\s+([\d.e+-]+)\s+"  # PSNR and Diff
                r"Success Ratio:\s+([\d.]+)\s+\|\s+Flip Count:\s+(\d+)\s+.*?"  # Success Ratio and Flip Count
                r"Reward :\s+([\d.e+-]+)\s+"  # Reward
                r"Time taken for this data:\s+([\d.]+)\s+seconds",  # Time taken
                content,
                re.DOTALL,
            )

            for match in step_matches:
                step = int(match.group(1))  # 스텝 값
                if step % step_unit == 0 and step <= max_step:
                    try:
                        psnr_diff = float(match.group(2))  # PSNR 차이값
                        success_ratio = float(match.group(3))  # 성공률
                        flip_count = int(match.group(4))  # Flip Count
                        reward = float(match.group(5))  # Reward 값
                        time_taken = float(match.group(6))  # 데이터 처리 시간

                        # 스텝별로 데이터를 그룹화
                        step_data[step].append({
                            "psnr_diff": psnr_diff,
                            "success_ratio": success_ratio,
                            "flip_count": flip_count,
                            "reward": reward,
                            "time_taken": time_taken,
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing entry: {match.group(0)}\n{e}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"File not found: {file_path}")
    except UnicodeDecodeError:
        messagebox.showerror("Error", "File encoding issue. Please ensure the file is UTF-8 encoded.")

    return step_data

# 표준편차, 표준오차 및 평균 계산
def calculate_stats(values):
    if not values:
        return None, None, None
    mean = np.mean(values)  # 평균
    std_dev = np.std(values, ddof=1)  # 표준편차
    std_err = std_dev / np.sqrt(len(values))  # 표준오차
    return mean, std_dev, std_err

# 파일 열기 및 처리
def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a log file",
        filetypes=[("Log files", "*.log"), ("Text files", "*.txt")]
    )
    if file_path:
        try:
            step_unit = int(step_unit_entry.get())
            max_step = int(max_step_entry.get())
            if step_unit <= 0 or max_step <= 0:
                raise ValueError("Step unit and max step must be positive integers.")

            step_data = parse_log_file(file_path, step_unit, max_step)
            if not step_data:
                messagebox.showinfo("Info", "No matching steps found in the log file.")
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "No data found in the log file.")
                return

            results = []
            for step, data_list in sorted(step_data.items()):
                psnr_diff_values = [d["psnr_diff"] for d in data_list]
                success_ratio_values = [d["success_ratio"] for d in data_list]
                reward_values = [d["reward"] for d in data_list]
                time_values = [d["time_taken"] for d in data_list]

                psnr_diff_mean, psnr_diff_std_dev, psnr_diff_std_err = calculate_stats(psnr_diff_values)
                success_ratio_mean, success_ratio_std_dev, success_ratio_std_err = calculate_stats(success_ratio_values)
                reward_mean, reward_std_dev, reward_std_err = calculate_stats(reward_values)
                time_mean, time_std_dev, time_std_err = calculate_stats(time_values)

                results.append(
                    f"Step: {step}\n"
                    f"  PSNR Diff - Mean: {psnr_diff_mean:.6f}, Standard Deviation: {psnr_diff_std_dev:.6f}, "
                    f"Standard Error: {psnr_diff_std_err:.6f}\n"
                    f"  Success Ratio - Mean: {success_ratio_mean:.6f}, Standard Deviation: {success_ratio_std_dev:.6f}, "
                    f"Standard Error: {success_ratio_std_err:.6f}\n"
                    f"  Reward - Mean: {reward_mean:.6f}, Standard Deviation: {reward_std_dev:.6f}, "
                    f"Standard Error: {reward_std_err:.6f}\n"
                    f"  Time - Mean: {time_mean:.6f}\n"
                )

            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, "\n".join(results))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# GUI 설정
root = tk.Tk()
root.title("Log File Step-wise Analysis")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

tk.Label(frame, text="Step Unit:").grid(row=0, column=0, sticky="e")
step_unit_entry = tk.Entry(frame)
step_unit_entry.grid(row=0, column=1)
step_unit_entry.insert(0, "100")

tk.Label(frame, text="Max Step:").grid(row=1, column=0, sticky="e")
max_step_entry = tk.Entry(frame)
max_step_entry.grid(row=1, column=1)
max_step_entry.insert(0, "3000")

tk.Button(frame, text="Select Log File", command=open_file).grid(row=2, column=0, columnspan=2, pady=10)

# 텍스트 위젯에 스크롤바 추가
result_frame = tk.Frame(root)
result_frame.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(result_frame)
scrollbar.pack(side="right", fill="y")

result_text = tk.Text(result_frame, wrap="word", height=20, width=92, padx=10, pady=10, yscrollcommand=scrollbar.set)
result_text.pack(side="left", fill="both", expand=True)
scrollbar.config(command=result_text.yview)

root.mainloop()