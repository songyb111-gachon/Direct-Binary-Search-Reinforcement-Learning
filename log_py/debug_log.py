import tkinter as tk
from tkinter import filedialog, simpledialog
import re
import pandas as pd


# 파일 선택 GUI
def select_log_file():
    root = tk.Tk()
    root.withdraw()  # GUI 창을 숨김
    file_path = filedialog.askopenfilename(filetypes=[("Log files", "*.log")])
    return file_path


# 스텝 범위 입력받기
def get_step_range(default_range="2-512"):
    root = tk.Tk()
    root.withdraw()  # GUI 창을 숨김
    step_range = simpledialog.askstring(
        "Input",
        "Enter step range (e.g., 2-512):",
        initialvalue=default_range  # 기본값 설정
    )
    return step_range


# 로그 파일 처리 및 평균 시간 계산
def process_log_file(file_path, start_step, end_step):
    # 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        log_data = file.readlines()

    # 데이터 정리
    step_data = []
    for line in log_data:
        # 로그 형식에 맞는 정규식
        match = re.search(
            r"Step:\s*(\d+)\s*\|\s*Time\s*(.+?)\s*:\s*([\d\.]+)\s*seconds", line
        )
        if match:
            step, action, time = match.groups()
            step_data.append({"Step": int(step), "Action": action.strip(), "Time": float(time)})

    # 데이터프레임 생성
    df = pd.DataFrame(step_data)

    if df.empty:
        print("No matching data found in the log file.")
        return None

    # 특정 스텝 범위 필터링
    filtered_df = df[(df['Step'] >= start_step) & (df['Step'] <= end_step)]
    if filtered_df.empty:
        print(f"No data found for the specified step range: {start_step}-{end_step}")
        return None

    # 평균 시간 계산 (스텝 범위 내 데이터 수 기준)
    avg_times = filtered_df.groupby('Action')['Time'].mean()
    counts = filtered_df.groupby('Action')['Step'].count()  # 범위 내 카운트 계산
    result = pd.DataFrame({'Average Time': avg_times, 'Count in Range': counts})

    # 소수점 형식으로 출력
    pd.set_option("display.float_format", "{:.6f}".format)
    return result


# 실행 코드
if __name__ == "__main__":
    file_path = select_log_file()
    if file_path:
        step_range = get_step_range(default_range="2-512")  # 기본값 전달
        if step_range:
            try:
                start_step, end_step = map(int, step_range.split('-'))
                avg_times = process_log_file(file_path, start_step, end_step)
                if avg_times is not None:
                    print("\nAverage times for actions in the step range:")
                    print(avg_times)
            except ValueError:
                print("Invalid step range format. Please use 'start-end' format (e.g., 338-400).")
        else:
            print("Step range not entered.")
    else:
        print("File not selected.")