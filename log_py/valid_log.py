import re
import tkinter as tk
from tkinter import filedialog, messagebox


def parse_log_data(log_data):
    """
    로그 데이터를 파싱하여 각 에피소드의 결과를 추출.
    """
    results = []
    # 에피소드 구분
    episodes = re.split(r"\[Episode Start\] Currently using dataset file: \((.+?)\), Episode count: \d+", log_data)

    if len(episodes) < 2:
        return ["No valid data found in the log file."]

    datasets = episodes[1::2]
    logs = episodes[2::2]

    for dataset, log in zip(datasets, logs):
        result = [f"Dataset: {dataset.strip()}"]

        # 스텝 데이터 추출
        step_matches = re.finditer(
            r"Step: (\d+)\s+\| Initial PSNR: ([\d.]+)\s+"
            r"PSNR After: ([\d.]+)\s+\|\s+Change: ([\d.e+-]+)\s+\|\s+Diff: ([\d.e+-]+)\s+"
            r"Reward: ([\d.]+)\s+\|\s+Success Ratio: ([\d.e+-]+)\s+\|\s+Flip Count: (\d+).*?"
            r"Time taken for this data: ([\d.]+) seconds",
            log,
            re.DOTALL,
        )

        for match in step_matches:
            step = int(match.group(1))
            initial_psnr = float(match.group(2))
            psnr_after = float(match.group(3))
            change = float(match.group(4))
            diff = float(match.group(5))
            reward = float(match.group(6))
            success_ratio = float(match.group(7))
            flip_count = int(match.group(8))
            time_taken = float(match.group(9))

            result.append(
                f"PSNR diff: {diff:.6f}\n"
                f"Step: {step}\n"
                f"Time: {time_taken:.2f}s\n"
                f"Success Ratio: {success_ratio:.6f}\n"
                f"Flip Count: {flip_count}\n"
                #f"Initial PSNR: {initial_psnr:.6f}\n"
                #f"PSNR After: {psnr_after:.6f}\n"
                #f"Change: {change:.6f}\n"
                #f"Reward: {reward:.2f}\n"
            )

        if len(result) == 1:
            result.append("No steps found in this dataset.")

        results.append("\n".join(result))
    return results


def open_log_file():
    """
    로그 파일을 열고 파싱한 결과를 GUI에 표시.
    """
    file_path = filedialog.askopenfilename(
        title="Select Log File",
        filetypes=[("Log Files", "*.log"), ("All Files", "*.*")]
    )
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a log file.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            log_data = file.read()

        parsed_results = parse_log_data(log_data)

        # 결과를 새 창에 표시
        result_window = tk.Toplevel()
        result_window.title("Parsed Log Results")
        text_widget = tk.Text(result_window, wrap="word")
        text_widget.pack(expand=True, fill="both")
        text_widget.insert("1.0", "\n\n".join(parsed_results))

        # 저장 버튼 추가
        save_button = tk.Button(result_window, text="Save Results", command=lambda: save_results(parsed_results))
        save_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while reading the log file:\n{str(e)}")


def save_results(results):
    """
    파싱한 결과를 파일로 저장.
    """
    save_path = filedialog.asksaveasfilename(
        title="Save Results As",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if save_path:
        with open(save_path, "w", encoding="utf-8") as file:
            file.write("\n\n".join(results))
        messagebox.showinfo("Saved", f"Results saved to {save_path}")


# GUI 설정
root = tk.Tk()
root.title("Log Parser Tool")
root.geometry("400x200")

label = tk.Label(root, text="Log Parser Tool", font=("Arial", 16))
label.pack(pady=20)

button = tk.Button(root, text="Open Log File", command=open_log_file, width=20, height=2)
button.pack(pady=20)

root.mainloop()