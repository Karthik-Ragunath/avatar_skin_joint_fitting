import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import re

# Sample log lines
# log_data = """
# 2023-10-30 07:07:15,439 | INFO | epoch - 1; mesh_index: 0; mini_batch_index: 0 - error: 0.34121379256248474
# 2023-10-30 07:07:17,227 | INFO | epoch - 1; mesh_index: 1; mini_batch_index: 0 - error: 0.330465704202652
# ...
# 2023-10-30 07:07:40,344 | INFO | epoch - 1; mesh_index: 14; mini_batch_index: 0 - error: 0.1847173273563385
# """

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", required=True, type=str)
parser.add_argument("--file_type", required=False, type=str, default="reconstruction")
parser.add_argument("--output_file_path", required=True, type=str)
args = parser.parse_args()

file_type = args.file_type

with open(args.file_path, 'r') as f:
    log_data = f.read()

# Parse log data to extract error values
if file_type == "reconstruction":
    errors = []
    for line in log_data.strip().split('\n'):
        try:
            error = float(line.split("error:")[1].strip())
        except Exception as _:
            continue
        errors.append(error)
else:
    recon_losses = []
    smoothing_losses = []
    for line in log_data.strip().split('\n'):
        try:
            recon_match = re.search(r'recon_loss: ([\d.]+)', line)
            smoothing_match = re.search(r'smoothing_loss: ([\d.]+)', line)
            
            if recon_match and smoothing_match:
                recon_losses.append(float(recon_match.group(1)))
                smoothing_losses.append(float(smoothing_match.group(1)))
        except Exception as _:
            continue

if file_type == "reconstruction":
    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o', linestyle='-', color='b', linewidth=0.4)
    plt.xlabel('Iterations')
    plt.ylabel('Error Value')
    plt.title('Error over Iterations')
    plt.grid(True)

    output_path = args.output_file_path
    directoy_name = Path(output_path).parent
    os.makedirs(directoy_name, exist_ok=True)
    plt.savefig(output_path)
else:
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Recon loss plot
    ax1.plot(recon_losses, label='Recon Loss', color='blue')
    ax1.set_title('Recon Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.legend()

    # Smoothing loss plot
    ax2.plot(smoothing_losses, label='Smoothing Loss', color='red')
    ax2.set_title('Smoothing Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Value')
    ax2.legend()

    plt.tight_layout()
    output_path = args.output_file_path
    directoy_name = Path(output_path).parent
    os.makedirs(directoy_name, exist_ok=True)
    plt.savefig(output_path)