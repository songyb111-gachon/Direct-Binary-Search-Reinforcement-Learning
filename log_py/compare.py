import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


def parse_log1(log_text):
    """
    첫 번째 로그(픽셀 플립 최적화 로그)를 파싱합니다.
    각 에피소드마다 파일 이름, 에피소드 번호, 초기 PSNR, 최종 PSNR 개선량,
    마지막 처리 시간, 그리고 첫 번째 등장하는 Step 값을 추출합니다.
    에피소드 정보가 없는 블록은 무시합니다.
    """
    episodes = []
    blocks = log_text.split("[Episode Start]")
    for block in blocks:
        block = block.strip()
        if not block or "Episode count:" not in block:
            continue

        # 에피소드 번호 추출 (예: Episode count: 1)
        ep_match = re.search(r"Episode count:\s*(\d+)", block)
        episode = int(ep_match.group(1)) if ep_match else None

        # 파일 이름 추출 (예: .../0001.png)
        file_match = re.search(r"Currently using dataset file:\s*\('.*?([^/]+\.png)',\)", block)
        file_name = file_match.group(1) if file_match else None

        # 초기 PSNR 추출 (필요에 따라 사용)
        init_match = re.search(r"Initial PSNR:\s*([\d.]+)", block)
        init_psnr = float(init_match.group(1)) if init_match else None

        # 최종 PSNR 개선량 추출 (필요에 따라 사용)
        improvement_match = re.search(r"Optimization completed\. Final PSNR improvement:\s*([\d.]+)", block)
        final_improvement = float(improvement_match.group(1)) if improvement_match else None

        # 마지막 처리 시간 추출 (에피소드 내 마지막 "Time taken for this data" 값)
        time_matches = re.findall(r"Time taken for this data:\s*([\d.]+) seconds", block)
        time_taken = float(time_matches[-1]) if time_matches else None

        # 첫 번째 Step 값 추출 (정수)
        step_match = re.search(r"Step:\s*(\d+)", block)
        step = int(step_match.group(1)) if step_match else None

        episodes.append({
            "episode": episode,
            "file": file_name,
            "init_psnr": init_psnr,
            "final_improvement": final_improvement,
            "time": time_taken,
            "step": step
        })
    return episodes


def parse_log2(log_text):
    """
    두 번째 로그(보상 및 에피소드별 로그)를 파싱합니다.
    각 에피소드마다 파일 이름, 에피소드 번호, 초기 PSNR, 총 보상,
    각 step의 평균 처리 시간, 그리고 첫 번째 등장하는 Step 값을 추출합니다.
    에피소드 정보가 없는 블록은 무시합니다.
    """
    episodes = []
    blocks = log_text.split("[Episode Start]")
    for block in blocks:
        block = block.strip()
        if not block or "Episode count:" not in block:
            continue

        ep_match = re.search(r"Episode count:\s*(\d+)", block)
        episode = int(ep_match.group(1)) if ep_match else None

        file_match = re.search(r"Currently using dataset file:\s*\('.*?([^/]+\.png)',\)", block)
        file_name = file_match.group(1) if file_match else None

        init_match = re.search(r"Initial PSNR:\s*([\d.]+)", block)
        init_psnr = float(init_match.group(1)) if init_match else None

        reward_match = re.search(r"Episode\s*\d+:\s*Total Reward:\s*([-\d.]+)", block)
        total_reward = float(reward_match.group(1)) if reward_match else None

        time_matches = re.findall(r"Time taken for this data:\s*([\d.]+) seconds", block)
        if time_matches:
            times = [float(t) for t in time_matches]
            avg_time = sum(times) / len(times)
        else:
            avg_time = None

        step_match = re.search(r"Step:\s*(\d+)", block)
        step = int(step_match.group(1)) if step_match else None

        episodes.append({
            "episode": episode,
            "file": file_name,
            "init_psnr": init_psnr,
            "total_reward": total_reward,
            "time": avg_time,
            "step": step
        })
    return episodes


def generate_comparison_texts(log1_text, log2_text):
    """
    두 로그의 텍스트를 파싱한 후, 각 에피소드별로
    로그1의 Step, 로그2의 Step, 그리고 그 차이(로그2 - 로그1)를 계산하여
    원래 데이터 순서(에피소드 순)와 차이값 순으로 정렬한 두 가지 결과를
    각각 표 형식의 문자열로 반환합니다.
    """
    log1_eps = parse_log1(log1_text)
    log2_eps = parse_log2(log2_text)

    # 두 로그의 에피소드별 데이터를 zip()을 사용하여 결합
    rows = []
    for ep1, ep2 in zip(log1_eps, log2_eps):
        step1 = ep1.get("step")
        step2 = ep2.get("step")
        diff = None
        if step1 is not None and step2 is not None:
            diff = step2 - step1
        rows.append({
            "episode": ep1.get("episode"),
            "file": ep1.get("file"),
            "step1": step1,
            "step2": step2,
            "diff": diff
        })

    # 표 헤더와 구분선
    header = f"{'Episode':>7} | {'File':>7} | {'Step (Log1)':>12} | {'Step (Log2)':>12} | {'Diff':>10}\n"
    divider = "-" * (len(header) - 1) + "\n"

    def safe_format_int(value, width):
        if value is None:
            return "N/A".rjust(width)
        else:
            return f"{value:>{width}d}"

    # (1) 원래 데이터(에피소드 순) 정렬 결과
    natural_lines = [header, divider]
    for row in rows:
        line = (f"{safe_format_int(row['episode'], 7)} | "
                f"{str(row['file']).rjust(7)} | "
                f"{safe_format_int(row['step1'], 12)} | "
                f"{safe_format_int(row['step2'], 12)} | "
                f"{safe_format_int(row['diff'], 10)}\n")
        natural_lines.append(line)
    natural_output = "".join(natural_lines)

    # (2) 차이값(diff)을 기준으로 오름차순 정렬한 결과
    sorted_rows = sorted(rows, key=lambda r: r['diff'] if r['diff'] is not None else float('inf'))
    sorted_lines = [header, divider]
    for row in sorted_rows:
        line = (f"{safe_format_int(row['episode'], 7)} | "
                f"{str(row['file']).rjust(7)} | "
                f"{safe_format_int(row['step1'], 12)} | "
                f"{safe_format_int(row['step2'], 12)} | "
                f"{safe_format_int(row['diff'], 10)}\n")
        sorted_lines.append(line)
    sorted_output = "".join(sorted_lines)

    return natural_output, sorted_output


def main():
    # Tkinter 기본 창 생성 후 숨기기
    root = tk.Tk()
    root.withdraw()

    # 첫 번째 로그 파일 선택
    messagebox.showinfo("파일 선택", "첫 번째 로그 파일 (예: log1.log)을 선택하세요.")
    log1_path = filedialog.askopenfilename(
        title="첫 번째 로그 파일 선택",
        filetypes=[("Log Files", "*.log"), ("All Files", "*.*")]
    )
    if not log1_path:
        messagebox.showerror("오류", "첫 번째 로그 파일이 선택되지 않았습니다.")
        return

    # 두 번째 로그 파일 선택
    messagebox.showinfo("파일 선택", "두 번째 로그 파일 (예: log2.log)을 선택하세요.")
    log2_path = filedialog.askopenfilename(
        title="두 번째 로그 파일 선택",
        filetypes=[("Log Files", "*.log"), ("All Files", "*.*")]
    )
    if not log2_path:
        messagebox.showerror("오류", "두 번째 로그 파일이 선택되지 않았습니다.")
        return

    try:
        with open(log1_path, "r", encoding="utf-8") as f:
            log1_text = f.read()
    except Exception as e:
        messagebox.showerror("오류", f"{log1_path} 파일 읽는 중 오류 발생:\n{e}")
        return

    try:
        with open(log2_path, "r", encoding="utf-8") as f:
            log2_text = f.read()
    except Exception as e:
        messagebox.showerror("오류", f"{log2_path} 파일 읽는 중 오류 발생:\n{e}")
        return

    # 두 로그에서 비교 결과(원래 순서 및 차이값 정렬)를 생성
    natural_output, sorted_output = generate_comparison_texts(log1_text, log2_text)
    combined_text = "==== 데이터 순서(에피소드 순) 정렬 결과 ====\n" + natural_output + "\n\n"
    combined_text += "==== 차이값(로그2 - 로그1) 순 정렬 결과 ====\n" + sorted_output

    # 결과를 표시할 새 창 생성 (스크롤 가능한 텍스트 위젯 사용)
    result_window = tk.Toplevel()
    result_window.title("Step 비교 결과")
    result_window.geometry("800x600")

    text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, font=("Consolas", 10))
    text_area.insert(tk.END, combined_text)
    text_area.configure(state='disabled')
    text_area.pack(expand=True, fill='both')

    root.mainloop()


if __name__ == "__main__":
    main()