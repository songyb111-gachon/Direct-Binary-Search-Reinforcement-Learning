import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
import numpy as np

def parse_and_plot_comparison(text):
    # Split input into two datasets based on a delimiter (e.g., 'DBS' and 'reinforcement')
    datasets = text.strip().split("\n\n")
    if len(datasets) != 2:
        messagebox.showerror("Error", "Please enter data for two datasets separated by a blank line.")
        return

    # Function to extract data using regex
    def extract_data(data_text):
        pattern = r"Range (\d\.\d-\d\.\d): Total Pixels = (\d+), Improved Pixels = (\d+), Improvement Ratio \(in range\) = ([\d\.]+), Improvement Ratio \(to total improved\) = ([\d\.]+), Total PSNR Improvement = ([\d\.]+), Average PSNR Improvement = ([\d\.]+)"
        matches = re.findall(pattern, data_text)

        if not matches:
            return None

        # Extract data
        ranges = []
        total_pixels = []
        improved_pixels = []
        improvement_ratio_in_range = []
        improvement_ratio_total = []
        total_psnr_improvement = []
        average_psnr_improvement = []

        for match in matches:
            ranges.append(match[0])
            total_pixels.append(int(match[1]))
            improved_pixels.append(int(match[2]))
            improvement_ratio_in_range.append(float(match[3]))
            improvement_ratio_total.append(float(match[4]))
            total_psnr_improvement.append(float(match[5]))
            average_psnr_improvement.append(float(match[6]))

        return {
            "ranges": ranges,
            "total_pixels": total_pixels,
            "improved_pixels": improved_pixels,
            "improvement_ratio_in_range": improvement_ratio_in_range,
            "improvement_ratio_total": improvement_ratio_total,
            "total_psnr_improvement": total_psnr_improvement,
            "average_psnr_improvement": average_psnr_improvement,
        }

    # Extract data for both datasets
    dataset1 = extract_data(datasets[0])
    dataset2 = extract_data(datasets[1])

    if not dataset1 or not dataset2:
        messagebox.showerror("Error", "Invalid data format in one or both datasets.")
        return

    # Convert ranges to indices for plotting
    x = np.arange(len(dataset1["ranges"]))
    bar_width = 0.35

    # Plot 1: Improved vs Total Pixels
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width, dataset1["improved_pixels"], bar_width, label="Improved Pixels (Dataset 1)")
    plt.bar(x, dataset1["total_pixels"], bar_width, label="Total Pixels (Dataset 1)")
    plt.bar(x + bar_width, dataset2["improved_pixels"], bar_width, label="Improved Pixels (Dataset 2)", alpha=0.7)
    plt.bar(x + 2 * bar_width, dataset2["total_pixels"], bar_width, label="Total Pixels (Dataset 2)", alpha=0.7)
    plt.xticks(x, dataset1["ranges"])
    plt.xlabel("Range")
    plt.ylabel("Pixel Count")
    plt.title("Improved vs Total Pixels Across Ranges (Comparison)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 2: Improvement Ratios
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width / 2, dataset1["improvement_ratio_in_range"], bar_width, label="Improvement Ratio (in range, Dataset 1)")
    plt.bar(x + bar_width / 2, dataset2["improvement_ratio_in_range"], bar_width, label="Improvement Ratio (in range, Dataset 2)")
    plt.xticks(x, dataset1["ranges"])
    plt.xlabel("Range")
    plt.ylabel("Improvement Ratio")
    plt.title("Improvement Ratios Across Ranges (Comparison)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 3: Total PSNR Improvement
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width / 2, dataset1["total_psnr_improvement"], bar_width, label="Total PSNR Improvement (Dataset 1)")
    plt.bar(x + bar_width / 2, dataset2["total_psnr_improvement"], bar_width, label="Total PSNR Improvement (Dataset 2)")
    plt.xticks(x, dataset1["ranges"])
    plt.xlabel("Range")
    plt.ylabel("Total PSNR Improvement (dB)")
    plt.title("Total PSNR Improvement Across Ranges (Comparison)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot 4: Average PSNR Improvement
    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width / 2, dataset1["average_psnr_improvement"], bar_width, label="Average PSNR Improvement (Dataset 1)")
    plt.bar(x + bar_width / 2, dataset2["average_psnr_improvement"], bar_width, label="Average PSNR Improvement (Dataset 2)")
    plt.xticks(x, dataset1["ranges"])
    plt.xlabel("Range")
    plt.ylabel("Average PSNR Improvement (dB)")
    plt.title("Average PSNR Improvement Across Ranges (Comparison)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def on_submit():
    input_text = text_area.get("1.0", tk.END).strip()
    if input_text:
        parse_and_plot_comparison(input_text)
    else:
        messagebox.showerror("Error", "Please enter some text data.")

# Create the GUI window
root = tk.Tk()
root.title("Dataset Comparison")
root.geometry("600x400")

# Label
label = tk.Label(root, text="Enter data for two datasets separated by a blank line:", font=("Arial", 12))
label.pack(pady=10)

# ScrolledText for input
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10), height=15)
text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Submit Button
submit_button = ttk.Button(root, text="Generate Comparison Graphs", command=on_submit)
submit_button.pack(pady=10)

# Run the GUI event loop
root.mainloop()
