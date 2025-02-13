import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
import numpy as np

def parse_text_and_plot(text):
    # Regex pattern to extract data
    pattern = r"Range (\d\.\d-\d\.\d): Total Pixels = (\d+), Improved Pixels = (\d+), Improvement Ratio \(in range\) = ([\d\.]+), Improvement Ratio \(to total improved\) = ([\d\.]+), Total PSNR Improvement = ([\d\.]+), Average PSNR Improvement = ([\d\.]+)"
    matches = re.findall(pattern, text)

    # Check if data is valid
    if not matches:
        messagebox.showerror("Error", "No valid data found in the input.")
        return

    # Data storage
    ranges = []
    total_pixels = []
    improved_pixels = []
    improvement_ratio_in_range = []
    improvement_ratio_total = []
    total_psnr_improvement = []
    average_psnr_improvement = []

    # Extract data
    for match in matches:
        ranges.append(match[0])
        total_pixels.append(int(match[1]))
        improved_pixels.append(int(match[2]))
        improvement_ratio_in_range.append(float(match[3]))
        improvement_ratio_total.append(float(match[4]))
        total_psnr_improvement.append(float(match[5]))
        average_psnr_improvement.append(float(match[6]))

    # Convert ranges to indices for plotting
    x = np.arange(len(ranges))
    bar_width = 0.35

    # Plot 1: Improved vs Total Pixels (Combined)
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width / 2, improved_pixels, bar_width, label="Improved Pixels")
    plt.bar(x + bar_width / 2, total_pixels, bar_width, label="Total Pixels")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Pixel Count")
    plt.title("Improved vs Total Pixels Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 1a: Improved Pixels (Individual)
    plt.figure(figsize=(14, 7))
    plt.bar(x, improved_pixels, bar_width, color='blue', label="Improved Pixels")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Improved Pixels")
    plt.title("Improved Pixels Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 1b: Total Pixels (Individual)
    plt.figure(figsize=(14, 7))
    plt.bar(x, total_pixels, bar_width, color='orange', label="Total Pixels")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Total Pixels")
    plt.title("Total Pixels Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2: Combined Improvement Ratios
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width / 2, improvement_ratio_in_range, bar_width, label="Improvement Ratio (in range)")
    plt.bar(x + bar_width / 2, improvement_ratio_total, bar_width, label="Improvement Ratio (to total improved)")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Improvement Ratio")
    plt.title("Combined Improvement Ratios Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2a: Improvement Ratio (in range) - Individual
    plt.figure(figsize=(14, 7))
    plt.bar(x, improvement_ratio_in_range, bar_width, color='blue', label="Improvement Ratio (in range)")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Improvement Ratio (in range)")
    plt.title("Improvement Ratio (in range) Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2b: Improvement Ratio (to total improved) - Individual
    plt.figure(figsize=(14, 7))
    plt.bar(x, improvement_ratio_total, bar_width, color='orange', label="Improvement Ratio (to total improved)")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Improvement Ratio (to total improved)")
    plt.title("Improvement Ratio (to total improved) Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 3: Total PSNR Improvement
    plt.figure(figsize=(14, 7))
    plt.bar(x, total_psnr_improvement, bar_width, color='green', label="Total PSNR Improvement")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Total PSNR Improvement (dB)")
    plt.title("Total PSNR Improvement Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 4: Average PSNR Improvement
    plt.figure(figsize=(14, 7))
    plt.bar(x, average_psnr_improvement, bar_width, color='purple', label="Average PSNR Improvement")
    plt.xticks(x, ranges)
    plt.xlabel("Range")
    plt.ylabel("Average PSNR Improvement (dB)")
    plt.title("Average PSNR Improvement Across Ranges")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def on_submit():
    input_text = text_area.get("1.0", tk.END).strip()
    if input_text:
        parse_text_and_plot(input_text)
    else:
        messagebox.showerror("Error", "Please enter some text data.")

# Create the GUI window
root = tk.Tk()
root.title("Text Input for Graphs")
root.geometry("600x400")

# Label
label = tk.Label(root, text="Enter your data below:", font=("Arial", 12))
label.pack(pady=10)

# ScrolledText for input
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10), height=15)
text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Submit Button
submit_button = ttk.Button(root, text="Generate Graph", command=on_submit)
submit_button.pack(pady=10)

# Run the GUI event loop
root.mainloop()