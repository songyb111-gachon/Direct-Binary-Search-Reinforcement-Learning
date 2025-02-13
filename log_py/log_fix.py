import tkinter as tk
from tkinter import ttk, scrolledtext
import re

def calculate():
    input_data = text_input.get("1.0", tk.END).strip().split('\n')
    output_data = ""
    total_pixels_sum = 0

    pattern = re.compile(r"Range (.*?): Total Pixels = (\d+), Improved Pixels = (\d+), Improvement Ratio \(in range\) = ([0-9.]+), Improvement Ratio \(to total improved\) = ([0-9.]+), Total PSNR Improvement = ([0-9.]+), Average PSNR Improvement = ([0-9.eE+-]+)")

    for line in input_data:
        if not line.strip():
            continue

        try:
            match = pattern.match(line)
            if match:
                range_info = match.group(1)
                total_pixels = int(match.group(2))
                improved_pixels = int(match.group(3))
                improvement_ratio_in_range = float(match.group(4))
                improvement_ratio_to_total_improved = float(match.group(5))
                psnr_total = float(match.group(6))
                psnr_avg = float(match.group(7))

                # 계산 수행
                actual_total_pixels = total_pixels - improved_pixels
                actual_improvement_ratio_in_range = improved_pixels / actual_total_pixels if actual_total_pixels != 0 else 0

                # 총 합계 계산
                total_pixels_sum += actual_total_pixels

                # 과학적 표기법 제거
                psnr_avg_str = f"{psnr_avg:.8f}".rstrip('0').rstrip('.')

                # 결과 작성
                output_data += (f"Range {range_info}: Total Pixels = {actual_total_pixels}, Improved Pixels = {improved_pixels}, "
                                f"Improvement Ratio (in range) = {actual_improvement_ratio_in_range:.6f}, "
                                f"Improvement Ratio (to total improved) = {improvement_ratio_to_total_improved:.6f}, "
                                f"Total PSNR Improvement = {psnr_total}, Average PSNR Improvement = {psnr_avg_str}\n")
            else:
                output_data += f"Error processing line: {line}\nError: Pattern did not match\n"

        except Exception as e:
            output_data += f"Error processing line: {line}\nError: {e}\n"

    # 총합 및 비교값 추가
    comparison_value = 896 * 896 * 24
    output_data += f"\nTotal of all Total Pixels: {total_pixels_sum}\n"
    output_data += f"896 * 896 * 24 = {comparison_value}\n"

    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, output_data)

# GUI setup
root = tk.Tk()
root.title("PSNR Improvement Calculator")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Input label and text box
label_input = ttk.Label(frame, text="Input Data:")
label_input.grid(row=0, column=0, sticky=tk.W)

text_input = scrolledtext.ScrolledText(frame, width=100, height=20)
text_input.grid(row=1, column=0, sticky=(tk.W, tk.E))

# Calculate button
btn_calculate = ttk.Button(frame, text="Calculate", command=calculate)
btn_calculate.grid(row=2, column=0, pady=10)

# Output label and text box
label_output = ttk.Label(frame, text="Output Data:")
label_output.grid(row=3, column=0, sticky=tk.W)

text_output = scrolledtext.ScrolledText(frame, width=100, height=20)
text_output.grid(row=4, column=0, sticky=(tk.W, tk.E))

root.mainloop()