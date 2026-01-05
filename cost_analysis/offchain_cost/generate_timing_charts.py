import json
import os
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# Load Times New Roman from bundled font directory to avoid fallback to DejaVu
font_dir = Path(__file__).resolve().parent.parent.parent / 'font'
if font_dir.is_dir():
    for font_path in sorted(font_dir.glob('*.ttf')):
        try:
            fm.fontManager.addfont(str(font_path))
        except Exception:
            pass

# Increase default font sizes for clearer exported charts with Times New Roman
plt.rcParams.update({
    'font.family': ['Times New Roman', 'serif'],
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 25,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 21,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def load_json_data(filename):
    """Load JSON data."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_data(filename):
    """Load CSV data into a list of row dictionaries."""
    rows = []
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_latest_file(script_dir: str, pattern: str) -> str | None:
    base = Path(script_dir)
    candidates = list(base.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])

def generate_timing_comparison_charts(json_filename):
    """Generate tenure (T_tenure) comparison charts (from frequency-experiment JSON)."""
    # Load data
    data = load_json_data(json_filename)
    
    # Extract frequency-group data
    frequency_groups = data['frequency_groups']
    frequencies = []
    window_fill_times = []
    decision_times = []
    
    # Extract data ordered by frequency
    for freq_key in sorted(frequency_groups.keys(), key=lambda x: int(x.split('_')[1])):
        freq_data = frequency_groups[freq_key]
        frequencies.append(freq_data['frequency'])
        window_fill_times.append(freq_data['window_fill_times']['avg_ms'])
        decision_times.append(freq_data['decision_times']['avg_ms'])
    
    # Create a single chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    
    # Set x-axis positions
    x_pos = np.arange(len(frequencies))
    
    # Line 1: window fill time
    line1 = ax.plot(x_pos, window_fill_times, marker='o', markersize=8, 
                    color='#1f77b4', linewidth=2, markeredgecolor='#1f77b4', 
                    markeredgewidth=1.5, markerfacecolor='#1f77b4', label='Window Fill Time')
    
    # Line 2: decision time
    line2 = ax.plot(x_pos, decision_times, marker='s', markersize=8, 
                    color='#d62728', linewidth=2, markeredgecolor='#d62728', 
                    markeredgewidth=1.5, markerfacecolor='#d62728', label='Decision Time')
    
    # Axis labels
    ax.set_xlabel('T_tenure', fontsize=25)
    ax.set_ylabel('Time (ms)', fontsize=25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{f}' for f in frequencies])
    ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Y-axis range
    all_times = window_fill_times + decision_times
    max_time = max(all_times)
    min_time = min(all_times)
    y_range = max_time - min_time
    ax.set_ylim(min_time - y_range * 0.1, max_time + y_range * 0.2)
    
    # Value labels: window fill time
    for i, value in enumerate(window_fill_times):
        ax.text(x_pos[i], value + y_range * 0.03,
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, 
                color='#1f77b4', fontweight='bold')
    
    # Value labels: decision time
    for i, value in enumerate(decision_times):
        ax.text(x_pos[i], value + y_range * 0.03,
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, 
                color='#d62728', fontweight='bold')
    
    # Legend
    ax.legend(fontsize=21, loc='best', framealpha=0.9)
    
    # Layout
    plt.tight_layout()
    
    # Add a global title
    # fig.suptitle('DDQN Timing Performance Comparison Across Frequency Groups', 
    #              fontsize=16, fontweight='bold', y=1.02)
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"timing_comparison_tenure_{timestamp}.pdf"
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300)
    
    print(f"Combined timing comparison chart saved: {save_path}")
    print(f"Chart includes:")
    print(f"   - Window Fill Time: {window_fill_times} ms")
    print(f"   - Decision Time: {decision_times} ms")
    print(f"   - Frequency Groups: {frequencies}")
    
    # Show chart
    plt.show()
    
    return filename


def generate_scale_timing_charts(csv_filename):
    """Generate timing comparison charts across CA scales (from scale_timing_summary.csv)."""
    rows = load_csv_data(csv_filename)

    scales = []
    window_fill_times = []
    decision_times = []

    for row in rows:
        try:
            scales.append(int(row['ca_scale']))
            window_fill_times.append(float(row['avg_window_fill_ms']))
            decision_times.append(float(row['avg_decision_ms']))
        except Exception:
            continue

    if not scales:
        raise ValueError(f"No usable data found in CSV: {csv_filename}")

    order = np.argsort(scales)
    scales = [scales[i] for i in order]
    window_fill_times = [window_fill_times[i] for i in order]
    decision_times = [decision_times[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    x_pos = np.arange(len(scales))

    ax.plot(
        x_pos,
        window_fill_times,
        marker='o',
        markersize=8,
        color='#1f77b4',
        linewidth=2,
        markeredgecolor='#1f77b4',
        markeredgewidth=1.5,
        markerfacecolor='#1f77b4',
        label='Window Fill Time',
    )

    ax.plot(
        x_pos,
        decision_times,
        marker='s',
        markersize=8,
        color='#d62728',
        linewidth=2,
        markeredgecolor='#d62728',
        markeredgewidth=1.5,
        markerfacecolor='#d62728',
        label='Decision Time',
    )

    ax.set_xlabel('M', fontsize=25)
    ax.set_ylabel('Time (ms)', fontsize=25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in scales])
    ax.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.5)

    all_times = window_fill_times + decision_times
    max_time = max(all_times)
    min_time = min(all_times)
    y_range = max_time - min_time
    ax.set_ylim(min_time - y_range * 0.1, max_time + y_range * 0.2)

    for i, value in enumerate(window_fill_times):
        ax.text(
            x_pos[i],
            value + y_range * 0.03,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=14,
            color='#1f77b4',
            fontweight='bold',
        )

    for i, value in enumerate(decision_times):
        ax.text(
            x_pos[i],
            value + y_range * 0.03,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=14,
            color='#d62728',
            fontweight='bold',
        )

    ax.legend(fontsize=21, loc='best', framealpha=0.9)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"timing_comparison_scale_{timestamp}.pdf"
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=300)

    print(f"Scale timing comparison chart saved: {save_path}")
    print(f"Chart includes:")
    print(f"   - Window Fill Time: {window_fill_times} ms")
    print(f"   - Decision Time: {decision_times} ms")
    print(f"   - CA Scales: {scales}")

    plt.show()

    return filename


def generate_scale_frequency_heatmaps(csv_filename):
    """Use heatmaps to show how CA scale and tenure (T_tenure) affect timing (from scale_frequency_timing_summary.csv)."""
    rows = load_csv_data(csv_filename)

    scales = set()
    freqs = set()
    window_map = {}
    decision_map = {}

    for row in rows:
        try:
            s = int(row['ca_scale'])
            f = int(row['frequency'])
            w = float(row['avg_window_fill_ms'])
            d = float(row['avg_decision_ms'])
        except Exception:
            continue

        scales.add(s)
        freqs.add(f)
        window_map[(s, f)] = w
        decision_map[(s, f)] = d

    scales = sorted(scales)
    freqs = sorted(freqs)

    if not scales or not freqs:
        raise ValueError(f"No usable data found in CSV: {csv_filename}")

    window_mat = np.full((len(freqs), len(scales)), np.nan, dtype=float)
    decision_mat = np.full((len(freqs), len(scales)), np.nan, dtype=float)

    for i, f in enumerate(freqs):
        for j, s in enumerate(scales):
            if (s, f) in window_map:
                window_mat[i, j] = window_map[(s, f)]
            if (s, f) in decision_map:
                decision_mat[i, j] = decision_map[(s, f)]

    window_masked = np.ma.masked_invalid(window_mat)
    decision_masked = np.ma.masked_invalid(decision_mat)

    stroke = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]

    fig0, ax0 = plt.subplots(1, 1, figsize=(8, 5.5), constrained_layout=True)
    im0 = ax0.imshow(window_masked, aspect='auto', origin='lower', cmap='Blues')
    ax0.set_title('Window Fill Time', fontsize=20)
    ax0.set_xlabel('M', fontsize=25)
    ax0.set_ylabel('T_tenure', fontsize=25)
    ax0.set_xticks(np.arange(len(scales)))
    ax0.set_xticklabels([str(s) for s in scales])
    ax0.set_yticks(np.arange(len(freqs)))
    ax0.set_yticklabels([str(f) for f in freqs])
    fig0.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='ms')

    for i in range(window_mat.shape[0]):
        for j in range(window_mat.shape[1]):
            v = window_mat[i, j]
            if not np.isfinite(v):
                continue
            t = ax0.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=12, color='white')
            t.set_path_effects(stroke)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5.5), constrained_layout=True)
    im1 = ax1.imshow(decision_masked, aspect='auto', origin='lower', cmap='Blues')
    ax1.set_title('Decision Time', fontsize=20)
    ax1.set_xlabel('M', fontsize=25)
    ax1.set_ylabel('T_tenure', fontsize=25)
    ax1.set_xticks(np.arange(len(scales)))
    ax1.set_xticklabels([str(s) for s in scales])
    ax1.set_yticks(np.arange(len(freqs)))
    ax1.set_yticklabels([str(f) for f in freqs])
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='ms')

    for i in range(decision_mat.shape[0]):
        for j in range(decision_mat.shape[1]):
            v = decision_mat[i, j]
            if not np.isfinite(v):
                continue
            t = ax1.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=12, color='white')
            t.set_path_effects(stroke)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    window_filename = f"timing_heatmap_window_fill_scale_tenure_{timestamp}.pdf"
    window_path = os.path.join(script_dir, window_filename)
    fig0.savefig(window_path, dpi=300)

    decision_filename = f"timing_heatmap_decision_scale_tenure_{timestamp}.pdf"
    decision_path = os.path.join(script_dir, decision_filename)
    fig1.savefig(decision_path, dpi=300)

    print(f"Window Fill heatmap saved: {window_path}")
    print(f"Decision heatmap saved: {decision_path}")
    print(f"M values: {scales}")
    print(f"T_tenure(frequency): {freqs}")

    plt.show()

    return {
        'window_fill': window_filename,
        'decision': decision_filename,
    }

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Select plotting mode:")
    print("1. Tenure (T_tenure) timing comparison (frequency_comparison_report_*.json)")
    print("2. CA scale timing comparison (scale_timing_summary.csv)")
    print("3. CA scale x tenure heatmaps (scale_frequency_timing_summary.csv)")

    try:
        choice = input("Enter choice (default 1): ").strip()
    except EOFError:
        choice = ""

    if not choice:
        choice = "1"

    try:
        if choice == "1":
            json_file = find_latest_file(script_dir, "frequency_comparison_report_*.json")
            if not json_file:
                raise FileNotFoundError("frequency_comparison_report_*.json not found")
            chart_filename = generate_timing_comparison_charts(json_file)
            print("Chart generation completed successfully!")
        elif choice == "2":
            csv_file = os.path.join(script_dir, "scale_timing_summary.csv")
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"File not found: {csv_file}")
            chart_filename = generate_scale_timing_charts(csv_file)
            print("Chart generation completed successfully!")
        elif choice == "3":
            csv_file = os.path.join(script_dir, "scale_frequency_timing_summary.csv")
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"File not found: {csv_file}")
            chart_filename = generate_scale_frequency_heatmaps(csv_file)
            print("Chart generation completed successfully!")
        else:
            print(f"Invalid choice: {choice}")

    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
