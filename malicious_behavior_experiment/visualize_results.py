#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizing Experiment Results
==============================

This script loads simulation data and generates visualization plots.
It can be used independently of the simulation script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import re
import pickle
import pandas as pd

def get_output_root() -> str:
    configured = os.environ.get("DRL_DPKI_OUTPUT_DIR")
    if configured:
        return configured
    legacy = "/mnt/data/wy2024"
    if os.path.isdir(legacy):
        return legacy
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "outputs")

# Set random seed for reproducibility
np.random.seed(42)

# Load custom fonts
current_dir = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(current_dir, '../font')
if os.path.exists(font_dir):
    font_files = fm.findSystemFonts(fontpaths=[font_dir])
    for font_file in font_files:
        try:
            fm.fontManager.addfont(font_file)
        except Exception as e:
            print(f"Cannot load font {font_file}: {e}")
else:
    print(f"Warning: Font directory not found: {font_dir}")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans CN']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def get_unified_na_colors(max_na_count=20):
    """
    Get unified color configuration for NAs.
    """
    colors = plt.cm.tab20(np.linspace(0, 1, max_na_count))
    return colors


def reorder_na_legend(ax, **legend_kwargs):
    """Sort legend entries so NA curves appear in index order."""
    handles, labels = ax.get_legend_handles_labels()
    seen_handles = {}
    for handle, label in zip(handles, labels):
        if label not in seen_handles:
            seen_handles[label] = handle

    na_entries = []
    other_entries = []
    pattern = re.compile(r'^CA(\d+)')

    for label, handle in seen_handles.items():
        match = pattern.match(label)
        if match:
            na_entries.append((int(match.group(1)), label, handle))
        else:
            other_entries.append((label, handle))

    na_entries.sort(key=lambda item: item[0])

    sorted_handles = [entry[2] for entry in na_entries]
    sorted_labels = [entry[1] for entry in na_entries]

    sorted_handles.extend(handle for label, handle in other_entries)
    sorted_labels.extend(label for label, handle in other_entries)

    if sorted_handles:
        ax.legend(sorted_handles, sorted_labels, **legend_kwargs)


def visualize_predictions(test_case, prediction, save_path=None):
    """
    Visualize prediction results (reputation evolution curve only).
    
    Args:
        test_case: Test case
        prediction: Prediction result
        save_path: Output path
    """
    # Check whether this is an attack scenario
    is_attack_scenario = 'attack_type' in test_case
    
    if is_attack_scenario:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        attack_type = test_case['attack_type']
        
        raw_data = test_case['raw_data']
        best_idx = prediction['best_na_idx']
        
        # Reputation evolution curve
        if 'window_data_list' in test_case and test_case['window_data_list']:
            ax.set_title(f'Reputation Evolution Over Time\n(Attack: {attack_type}, NA Count: {test_case["n_na"]})', fontsize=14)
            ax.set_xlabel('Transaction Time')
            ax.set_ylabel('Reputation')
            
            # Plot reputation evolution for each NA using a unified tab20 colormap
            colors = plt.cm.tab20(np.linspace(0, 1, len(test_case['window_data_list'])))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'd', '|', '_', '1', '2', '3', '4', '8']
            
            for na_idx, window_data in enumerate(test_case['window_data_list']):
                # Compute reputation at each time point
                initial_reputation = raw_data[na_idx, 0]
                reputation_timeline = [initial_reputation]
                time_points = [0]  # Starting time point
                
                current_reputation = initial_reputation
                for pack in window_data:
                    # End time point of each pack
                    time_points.append(pack['end_step'] + 1)
                    current_reputation += pack['total_reputation_change']
                    reputation_timeline.append(current_reputation)
                
                # Line style rule: selected NA uses solid line, others use dashed lines
                is_selected = na_idx == best_idx  # Here, best_idx is the selected NA
                line_style = '-' if is_selected else '--'
                line_width = 3 if is_selected else 1.5
                alpha = 1.0 if is_selected else 0.6
                
                label = f'NA{na_idx}' + (' (Selected)' if is_selected else '')
                ax.plot(time_points, reputation_timeline, 
                       linestyle=line_style, linewidth=line_width, alpha=alpha,
                       color=colors[na_idx], label=label, marker=markers[na_idx % len(markers)], markersize=4)
            
            # Mark the historical window end time
            if test_case['window_data_list']:
                window_end_time = max([pack['end_step'] for pack in test_case['window_data_list'][0]]) + 1
                ax.axvline(x=window_end_time, color='red', linestyle=':', linewidth=2, 
                          label=f'Window End (t={window_end_time})')
                ax.text(window_end_time + 1, ax.get_ylim()[1] * 0.9, 
                       'Historical Window\nEnd Point', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=9, ha='left')
            
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Set y-axis range
            all_reputations = []
            for window_data in test_case['window_data_list']:
                initial_rep = raw_data[test_case['window_data_list'].index(window_data), 0]
                current_rep = initial_rep
                all_reputations.append(current_rep)
                for pack in window_data:
                    current_rep += pack['total_reputation_change']
                    all_reputations.append(current_rep)
            
            y_min, y_max = min(all_reputations), max(all_reputations)
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
    else:
        # For non-attack scenarios, show a simple message
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'Reputation Evolution Over Time\n(Only available for attack scenarios)', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title(f'DQN Model Prediction Analysis (NA Count: {test_case["n_na"]})', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization result saved: {save_path}")
    
    plt.show()

def visualize_na_metrics_evolution(evolution_data, save_path=None):
    """
    Visualize weighted success rate, weighted delay grade, and hunger level evolution for each NA
    
    Args:
        evolution_data: Three-phase evolution data
        save_path: Save path for the chart
    """
    plt.figure(figsize=(18, 12))
    
    # Create subplots for three metrics
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Get phase data
    phase2_data = evolution_data.get('phase2', {})
    phase3_data = evolution_data.get('phase3', {})
    attack_type = evolution_data.get('attack_type', 'Unknown')
    separator2 = evolution_data.get('phase_separator_2', 100)
    
    # Use a unified color configuration based on original NA indices
    phase2_selected = phase2_data.get('selected_indices', [])
    phase3_selected = phase3_data.get('selected_indices', [])
    unified_colors = get_unified_na_colors(20)
    na_colors = [unified_colors[i] for i in range(10)]
    
    # Plot 1: Weighted Success Rate
    ax1 = axes[0]
    
    # Phase 2 weighted success rates
    if 'weighted_success_rates' in phase2_data:
        for i in range(len(phase2_selected)):
            if i >= len(phase2_data['weighted_success_rates'][0]): continue
            weighted_rates = [rates[i] for rates in phase2_data['weighted_success_rates']]
            label = phase2_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase2_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            if 'malicious' in label.lower():
                ax1.plot(phase2_data['time_points'], weighted_rates, 
                        color=na_color, linewidth=2.5, marker='x', markersize=6,
                        markeredgecolor='red', markerfacecolor='red', markeredgewidth=2,
                        label=f'{label}', linestyle='-')
            else:
                ax1.plot(phase2_data['time_points'], weighted_rates, 
                        color=na_color, linewidth=2, marker='o', markersize=4,
                        label=label, linestyle='-')
    
    # Phase 3 weighted success rates
    if 'weighted_success_rates' in phase3_data:
        for i in range(len(phase3_selected)):
            if i >= len(phase3_data['weighted_success_rates'][0]): continue
            weighted_rates = [rates[i] for rates in phase3_data['weighted_success_rates']]
            label = phase3_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase3_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            if 'malicious' in label.lower():
                ax1.plot(phase3_data['time_points'], weighted_rates, 
                        color=na_color, linewidth=2.5, marker='x', markersize=6,
                        markeredgecolor='red', markerfacecolor='red', markeredgewidth=2,
                        label=f'{label}', linestyle='-')
            else:
                ax1.plot(phase3_data['time_points'], weighted_rates, 
                        color=na_color, linewidth=2, marker='^', markersize=4,
                        label=label, linestyle='-', alpha=0.8)
    
    ax1.axvline(x=separator2, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('Weighted Success Rate', fontsize=12)
    ax1.set_title(f'{attack_type.upper()} Attack Scenario - Weighted Success Rate Evolution', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Weighted Delay Grade
    ax2 = axes[1]
    
    # Phase 2 weighted delay grades
    if 'weighted_delay_grades' in phase2_data:
        for i in range(len(phase2_selected)):
            if i >= len(phase2_data['weighted_delay_grades'][0]): continue
            weighted_delays = [delays[i] for delays in phase2_data['weighted_delay_grades']]
            label = phase2_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase2_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            if 'malicious' in label.lower():
                ax2.plot(phase2_data['time_points'], weighted_delays, 
                        color=na_color, linewidth=2.5, marker='x', markersize=6,
                        markeredgecolor='red', markerfacecolor='red', markeredgewidth=2,
                        label=f'{label}', linestyle='-')
            else:
                ax2.plot(phase2_data['time_points'], weighted_delays, 
                        color=na_color, linewidth=2, marker='o', markersize=4,
                        label=label, linestyle='-')
    
    # Phase 3 weighted delay grades
    if 'weighted_delay_grades' in phase3_data:
        for i in range(len(phase3_selected)):
            if i >= len(phase3_data['weighted_delay_grades'][0]): continue
            weighted_delays = [delays[i] for delays in phase3_data['weighted_delay_grades']]
            label = phase3_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase3_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            if 'malicious' in label.lower():
                ax2.plot(phase3_data['time_points'], weighted_delays, 
                        color=na_color, linewidth=2.5, marker='x', markersize=6,
                        markeredgecolor='red', markerfacecolor='red', markeredgewidth=2,
                        label=f'{label}', linestyle='-')
            else:
                ax2.plot(phase3_data['time_points'], weighted_delays, 
                        color=na_color, linewidth=2, marker='^', markersize=4,
                        label=label, linestyle='-', alpha=0.8)
    
    ax2.axvline(x=separator2, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel('Weighted Delay Grade', fontsize=12)
    ax2.set_title('Weighted Delay Grade Evolution', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Plot 3: Hunger Level
    ax3 = axes[2]
    
    # Phase 2 hunger levels
    if 'hunger_levels' in phase2_data:
        for i in range(len(phase2_selected)):
            if i >= len(phase2_data['hunger_levels'][0]): continue
            hunger_levels = [levels[i] for levels in phase2_data['hunger_levels']]
            label = phase2_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase2_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            if 'malicious' in label.lower():
                ax3.plot(phase2_data['time_points'], hunger_levels, 
                        color=na_color, linewidth=2.5, marker='x', markersize=6,
                        markeredgecolor='red', markerfacecolor='red', markeredgewidth=2,
                        label=f'{label}', linestyle='-')
            else:
                ax3.plot(phase2_data['time_points'], hunger_levels, 
                        color=na_color, linewidth=2, marker='o', markersize=4,
                        label=label, linestyle='-')
    
    # Phase 3 hunger levels
    if 'hunger_levels' in phase3_data:
        for i in range(len(phase3_selected)):
            if i >= len(phase3_data['hunger_levels'][0]): continue
            hunger_levels = [levels[i] for levels in phase3_data['hunger_levels']]
            label = phase3_data['na_labels'][i].replace('NA', 'Node')
            original_na_idx = phase3_selected[i]  # Original NA index
            na_color = na_colors[original_na_idx]  # Unified color mapping
            
            # In phase 3, malicious behavior is disabled; use the normal marker for all NAs
            ax3.plot(phase3_data['time_points'], hungers, 
                    color=na_color, linewidth=2, marker='^', markersize=4,
                    label=label, linestyle='-', alpha=0.8)
    
    ax3.axvline(x=separator2, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Time Point', fontsize=12)
    ax3.set_ylabel('Hunger Level', fontsize=12)
    ax3.set_title('Hunger Level Evolution', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    
    # Add phase separator labels
    for ax in axes:
        ax.text(separator2 + 2, ax.get_ylim()[1] * 0.9, 'Phase 2→3 Transition', 
                rotation=90, verticalalignment='top', 
                color='green', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[CHART] NA Metrics Evolution chart saved: {save_path}")
    
    plt.show()


def visualize_transaction_evolution(evolution_data, save_path=None):
    """
    Visualize the 3-phase transaction simulation reputation and metrics evolution.
    
    Args:
        evolution_data: Dictionary containing evolution data
        save_path: Path to save the plot
    """
    should_show = save_path is None

    # Prepare shared data
    unified_colors = get_unified_na_colors(20)
    na_colors = [unified_colors[i] for i in range(10)]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'd', '|', '_', '1', '2', '3', '4', '8']
    
    phase1_data = evolution_data.get('phase1', {})
    phase2_data = evolution_data.get('phase2', {})
    phase3_data = evolution_data.get('phase3', {})
    attack_type = evolution_data.get('attack_type', 'Unknown')
    attack_mode = evolution_data.get('attack_mode', 'behavior')

    primary_metric_key = 'delay_grades' if attack_mode == 'delay_only' else 'reputations'
    primary_ylabel = 'Delay Level' if attack_mode == 'delay_only' else 'Reputation Value'
    
    separator1 = evolution_data.get('phase_separator_1', 50)
    separator2 = evolution_data.get('phase_separator_2', 100)
    
    phase2_selected = phase2_data.get('selected_indices', [])
    phase3_selected = phase3_data.get('selected_indices', [])
    all_selected = np.unique(np.concatenate([phase2_selected, phase3_selected])) if len(phase2_selected) > 0 or len(phase3_selected) > 0 else []

    # --- Reputation Evolution Plot ---
    with plt.rc_context({'font.family': 'Times New Roman'}):
        fig_reputation, ax_reputation = plt.subplots(figsize=(20, 10))

    legend_labels = set()

    def _label_for_na(na_idx):
        base_label = f'CA{na_idx}'
        if base_label in legend_labels:
            return f'_{base_label}'
        legend_labels.add(base_label)
        return base_label

    if attack_mode == 'delay_only' and 'all_delay_grade_data' in evolution_data:
        phase1_time_points = list(phase1_data.get('time_points', range(51)))
        phase2_time_points = list(phase2_data.get('time_points', []))
        phase3_time_points = list(phase3_data.get('time_points', []))

        sparse_time_points = list(phase1_time_points)
        if phase2_time_points:
            sparse_time_points.extend(phase2_time_points[1:])
        if phase3_time_points:
            sparse_time_points.extend(phase3_time_points[1:])

        full_time_points = list(range(0, 151))

        phase1_len = len(phase1_time_points)
        phase2_len = len(phase2_time_points)
        phase3_len = len(phase3_time_points)

        filled_series = {}

        phase2_selected_set = set(int(x) for x in phase2_selected) if len(phase2_selected) else set()
        phase3_selected_set = set(int(x) for x in phase3_selected) if len(phase3_selected) else set()

        for i in range(10):
            timeline = evolution_data['all_delay_grade_data'][i]

            y_values = list(timeline[:phase1_len])
            if phase2_len:
                y_values.extend(list(timeline[phase1_len:phase1_len + phase2_len])[1:])
            if phase3_len:
                start = phase1_len + phase2_len
                y_values.extend(list(timeline[start:start + phase3_len])[1:])

            filled = np.full(len(full_time_points), np.nan)
            for t_idx, t in enumerate(sparse_time_points):
                if 0 <= t < len(filled) and t_idx < len(y_values):
                    filled[t] = y_values[t_idx]

            last_value = filled[0]
            for t in range(len(filled)):
                if not np.isnan(filled[t]):
                    last_value = filled[t]
                else:
                    filled[t] = last_value

            filled_series[i] = filled

            masked = filled.copy()
            if i not in phase2_selected_set:
                masked[51:101] = np.nan
            if i not in phase3_selected_set:
                masked[101:151] = np.nan

            ax_reputation.plot(
                full_time_points,
                masked.tolist(),
                color=na_colors[i],
                linewidth=3,
                alpha=0.8,
                linestyle='-',
                marker=markers[i],
                markersize=10,
                markevery=5,
                label=_label_for_na(i)
            )

        malicious_original_indices = []
        for idx_in_selected in phase2_data.get('malicious_na_indices', []):
            if idx_in_selected < len(phase2_selected):
                malicious_original_indices.append(int(phase2_selected[idx_in_selected]))

        if malicious_original_indices and phase2_time_points:
            x_marks = [t for t in phase2_time_points if 0 <= t <= 150]
            for orig_idx in sorted(set(malicious_original_indices)):
                if 0 <= orig_idx < 10 and orig_idx in filled_series:
                    y_marks = [float(filled_series[orig_idx][t]) for t in x_marks]
                    ax_reputation.plot(
                        x_marks,
                        y_marks,
                        linestyle='None',
                        marker='x',
                        markersize=18,
                        markeredgecolor='red',
                        markerfacecolor='none',
                        markeredgewidth=3,
                        color=na_colors[orig_idx],
                        label=f"_{_label_for_na(orig_idx)}"
                    )

        ax_reputation.set_xlim(0, 150)
        ax_reputation.set_xticks([0, 50, 100, 150])
    else:
        if primary_metric_key in phase1_data:
            for i in range(10):
                y_values = [row[i] for row in phase1_data[primary_metric_key]]
                time_points = phase1_data.get('time_points', range(len(y_values)))
                na_color = na_colors[i]
                ax_reputation.plot(
                    time_points,
                    y_values,
                    color=na_color,
                    linewidth=3,
                    alpha=0.8,
                    linestyle='-',
                    marker=markers[i],
                    markersize=10,
                    markevery=5,
                    label=_label_for_na(i)
                )

        if primary_metric_key in phase2_data:
            for i in range(len(phase2_selected)):
                y_values = [row[i] for row in phase2_data[primary_metric_key]]
                original_na_idx = phase2_selected[i]
                na_color = na_colors[original_na_idx]
                malicious_indices = phase2_data.get('malicious_na_indices', [])
                
                if i in malicious_indices:
                    ax_reputation.plot(
                        phase2_data['time_points'],
                        y_values,
                        color=na_color,
                        linewidth=3,
                        marker='x',
                        markersize=18,
                        markeredgecolor='red',
                        markerfacecolor='none',
                        markeredgewidth=3,
                        label=_label_for_na(original_na_idx),
                        linestyle='-'
                    )
                else:
                    ax_reputation.plot(
                        phase2_data['time_points'],
                        y_values,
                        color=na_color,
                        linewidth=3,
                        label=_label_for_na(original_na_idx),
                        linestyle='-',
                        marker=markers[original_na_idx],
                        markersize=10
                    )

        if primary_metric_key in phase3_data:
            for i in range(len(phase3_selected)):
                y_values = [row[i] for row in phase3_data[primary_metric_key]]
                original_na_idx = phase3_selected[i]
                na_color = na_colors[original_na_idx]
                ax_reputation.plot(
                    phase3_data['time_points'],
                    y_values,
                    color=na_color,
                    linewidth=3,
                    label=_label_for_na(original_na_idx),
                    linestyle='-',
                    marker=markers[original_na_idx],
                    markersize=10
                )

    # Annotations
    ax_reputation.plot([], [], color='red', marker='x', linestyle='None', markersize=18,
                       markeredgewidth=3, markerfacecolor='none', label='Malicious CA')
    ax_reputation.axvline(x=separator1, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
    ax_reputation.axvline(x=separator2, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
    if attack_mode != 'delay_only':
        ax_reputation.axhline(y=6600, color='orange', linestyle='--', alpha=0.6, linewidth=2.5,
                              label='Reputation Threshold')
    ax_reputation.set_xlabel('Time Point', fontsize=80)
    ax_reputation.set_ylabel(primary_ylabel, fontsize=80)
    ax_reputation.tick_params(axis='both', labelsize=74)
    if attack_mode == 'delay_only':
        ax_reputation.set_ylim(-0.05, 1.05)
    reorder_na_legend(ax_reputation, loc='upper left', frameon=True, fontsize=25, ncol=1)
    ax_reputation.grid(True, alpha=0.3)

    fig_reputation.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        ext = '.pdf'
        reputation_save_path = f"{base}_reputation{ext}"
        fig_reputation.savefig(reputation_save_path, dpi=300, bbox_inches='tight')
        print(f"[CHART] Reputation evolution chart saved: {reputation_save_path}")
        plt.close(fig_reputation)
    elif should_show:
        plt.show()

    # Call the metrics visualization
    if save_path:
        base, ext = os.path.splitext(save_path)
        metrics_save_path = f"{base}_metrics.png"
        visualize_na_metrics_evolution(evolution_data, metrics_save_path)
    else:
        visualize_na_metrics_evolution(evolution_data)



def reconstruct_evolution_data_from_csv(folder_path, attack_type='ME'):
    """
    Attempt to reconstruct partial evolution data from CSV files.
    """
    evolution_data = {'attack_type': attack_type}
    
    # Define potential paths for CSV files
    potential_paths = [
        os.path.join(folder_path, f'{attack_type}_phase1_reputation_evolution.csv'),
        os.path.join(folder_path, 'experiment_code_data', f'{attack_type}_phase1_reputation_evolution.csv')
    ]
    
    phase1_file = None
    for path in potential_paths:
        if os.path.exists(path):
            phase1_file = path
            break
            
    if phase1_file:
        df = pd.read_csv(phase1_file)
        evolution_data['phase1'] = {}
        evolution_data['phase1']['time_points'] = df['time_step'].tolist()
        
        # Extract reputations for each NA
        reputations_list = []
        for i, row in df.iterrows():
            row_reps = []
            for j in range(10): # Assuming 10 NAs
                col_name = f'NA{j}_reputation'
                if col_name in df.columns:
                    row_reps.append(row[col_name])
                else:
                    row_reps.append(0)
            reputations_list.append(row_reps)
        evolution_data['phase1']['reputations'] = reputations_list
        print(f"Loaded Phase 1 data from {phase1_file}")
    
    # Note: Phase 2 and 3 CSVs (selection parameters) do not contain time-series evolution.
    # So we cannot reconstruct full evolution_data just from CSVs.
    
    return evolution_data


def main():
    print("Visualizing Results from Saved Data")
    
    base_dir = os.path.join(get_output_root(), "malicious_behavior_experiment")

    print("\n" + "="*40)
    print("Select Data Mode")
    print("="*40)
    print("1. behavior (existing)")
    print("2. delay_only (only delay manipulation)")

    mode_choice = input("\nPlease select mode (1-2): ").strip()
    attack_mode = 'delay_only' if mode_choice == '2' else 'behavior'

    if attack_mode == 'delay_only':
        data_dir = os.path.join(base_dir, 'experiment_code_data_delay_only')
    else:
        data_dir = base_dir

    print("\n" + "="*40)
    print("Select Experiment to Visualize")
    print("="*40)
    print("1. Malicious With Everyone (ME)")
    print("2. On-Off Attack (OOA)")
    print("3. Opportunistic Service Attack (OSA)")
    
    choice = input("\nPlease select experiment (1-3): ").strip()
    
    if choice == '1':
        attack_type = 'ME'
    elif choice == '2':
        attack_type = 'OOA'
    elif choice == '3':
        attack_type = 'OSA'
    else:
        print("Invalid choice, defaulting to ME")
        attack_type = 'ME'
    
    print(f"\nVisualizing results for: {attack_type}")
    
    # 1. Try to load full pickle data
    if attack_mode == 'delay_only':
        candidate_paths = [
            os.path.join(data_dir, f"{attack_type}_evolution_data.pkl"),
        ]
    else:
        candidate_paths = [
            os.path.join(data_dir, f"{attack_type}_evolution_data.pkl"),
            os.path.join(data_dir, "experiment_code_data", f"{attack_type}_evolution_data.pkl"),
        ]

    pkl_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            pkl_path = candidate
            break

    if pkl_path is not None:
        print(f"Loading full simulation data from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            evolution_data = pickle.load(f)
        
        visualize_transaction_evolution(evolution_data, os.path.join(data_dir, f"{attack_type}_visualize_results.png"))
    else:
        print(f"Full simulation data ({pkl_path}) not found.")
        print("Attempting to visualize from available CSV files...")
        
        evolution_data = reconstruct_evolution_data_from_csv(data_dir, attack_type)
        if 'phase1' in evolution_data:
            visualize_transaction_evolution(evolution_data, os.path.join(data_dir, f"{attack_type}_visualize_results_partial.png"))
        else:
            print("No sufficient data found to visualize.")

if __name__ == "__main__":
    main()
