#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment Result Plotting Tool
=====================================

Features:
- Read experiment-result JSON files produced by main.py
- Generate comparison charts for strategies
- Generate performance summary tables

Usage:
    python plot_results.py --file results/comparison_results.json --output results/plots
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.font_manager as fm

# Font loading configuration
# Prefer project-local fonts to avoid missing-font warnings on servers
font_dir = Path(__file__).resolve().parent.parent / 'font'
loaded_custom_font = False
if font_dir.is_dir():
    font_files = sorted(font_dir.glob('*.ttf'))
    for font_path in font_files:
        try:
            fm.fontManager.addfont(str(font_path))
            loaded_custom_font = True
        except Exception as exc:
            print(f"Font load failed {font_path.name}: {exc}")

# Default font family
preferred_fonts = ['Times New Roman', 'Times', 'serif'] if loaded_custom_font else ['DejaVu Sans', 'Liberation Sans', 'serif']
plt.rcParams['font.family'] = preferred_fonts
plt.rcParams['axes.unicode_minus'] = False

# Add the project path for optional config imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ResultPlotter:
    def __init__(self, result_file):
        self.result_file = Path(result_file)
        with open(self.result_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Read config info from data when available
        self.test_info = self.data.get('test_info', {})
        self.n_time_points = self.test_info.get('n_time_points', 100)
        self.n_na = self.test_info.get('n_na', 50)
        self.seed = self.test_info.get('seed', 42)

    def plot_all(self, output_dir):
        """Interactively generate charts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        available_actions = []

        available_actions.append({
            'name': 'Comparison Charts',
            'func': self._plot_comparison_charts,
            'args': (self.data.get('strategies', {}), self.data.get('malicious_counts', []), output_dir, self.n_na)
        })

        available_actions.append({
             'name': 'DRL Cold/Warm Comparison Charts',
             'func': self._plot_drl_cold_warm_charts,
             'args': (self.data.get('results', {}), self.data.get('malicious_counts', []), output_dir, self.n_na)
        })
        
        available_actions.append({
            'name': 'CA Scale Sensitivity Charts',
            'func': self._plot_ca_scale_sensitivity_charts,
            'args': (self.data, output_dir)
        })

        available_actions.append({
            'name': 'DRL Slide Window Ablation Charts',
            'func': self._plot_drl_slidewindow_ablation_charts,
            'args': (self.data.get('strategies', {}), self.data.get('malicious_counts', []), output_dir, self.n_na)
        })
        
        if not available_actions:
            print("Warning: Unrecognized data format; cannot generate charts.")
            return

        print("\n=== Available Chart Types ===")
        for i, action in enumerate(available_actions):
            print(f"{i + 1}. {action['name']}")
        print(f"{len(available_actions) + 1}. Generate All")
        print("q. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (number), or q to exit: ").strip().lower()
                if not choice:
                    continue
                
                if choice == 'q':
                    print("Exited.")
                    return
                
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(available_actions):
                    action = available_actions[choice_idx - 1]
                    print(f"\nGenerating: {action['name']}...")
                    action['func'](*action['args'])
                    print("Done.")
                elif choice_idx == len(available_actions) + 1:
                    print("\nGenerating all charts...")
                    for action in available_actions:
                        action['func'](*action['args'])
                    print("All done.")
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Enter a valid number, or q to exit.")
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                return

    def _plot_ca_scale_sensitivity_charts(self, sensitivity_data, output_dir):
        test_info = sensitivity_data.get('test_info', {})
        scales = test_info.get('scales', [400, 1000, 2500, 5200, 6000])
        malicious_ratios = test_info.get('malicious_ratios', list(range(0, 55, 5)))

        results = sensitivity_data.get('results', {})
        if not results:
            print("Error: Missing CA scale sensitivity experiment data.")
            return False

        axis_configs = {
            'malicious_na_selection_rate': {
                'title': '',
                'ylabel': 'Malicious CA\nSelection Rate (%)'
            },
            'avg_delay_level': {
                'title': '',
                'ylabel': 'Delay Rate (%)'
            },
            'reputation_change': {
                'title': '',
                'ylabel': 'Reputation Delta'
            },
            'avg_success_rate': {
                'title': '',
                'ylabel': 'Success Rate (%)'
            }
        }

        label_fontsize = 34
        tick_fontsize = 24
        title_fontsize = 30
        legend_fontsize = 24

        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'font.size': label_fontsize,
            'axes.titlesize': title_fontsize,
            'axes.labelsize': label_fontsize,
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'legend.fontsize': legend_fontsize,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        generated_any = False

        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']

        for metric_key, axis_cfg in axis_configs.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(axis_cfg['title'])
            ax.set_ylabel(axis_cfg['ylabel'])
            ax.grid(True, linestyle='--', alpha=0.3)

            legend_handles = {}

            for idx, n_na in enumerate(scales):
                scale_key = str(n_na)
                scale_data = results.get(scale_key, {})
                metrics_list = scale_data.get('metrics', [])
                if not metrics_list:
                    continue

                y_values = []
                ci_values = None

                ci_list = scale_data.get('metrics_ci95')
                if isinstance(ci_list, list) and len(ci_list) == len(metrics_list):
                    ci_values = []

                for idx, metrics in enumerate(metrics_list):
                    if metrics is None:
                        y_values.append(np.nan)
                        if ci_values is not None:
                            ci_values.append(np.nan)
                        continue

                    if metric_key == 'malicious_na_selection_rate':
                        y_values.append(metrics['malicious_na_selection_rate'] * 100)
                    elif metric_key == 'avg_delay_level':
                        y_values.append(metrics['avg_delay_level'])
                    elif metric_key == 'reputation_change':
                        y_values.append(metrics['reputation_change'])
                    elif metric_key == 'avg_success_rate':
                        y_values.append(metrics['avg_success_rate'] * 100)

                    if ci_values is not None:
                        ci_metrics = ci_list[idx]
                        if ci_metrics is None:
                            ci_values.append(np.nan)
                        else:
                            if metric_key == 'malicious_na_selection_rate':
                                ci_values.append(ci_metrics['malicious_na_selection_rate'] * 100)
                            elif metric_key == 'avg_delay_level':
                                ci_values.append(ci_metrics['avg_delay_level'])
                            elif metric_key == 'reputation_change':
                                ci_values.append(ci_metrics['reputation_change'])
                            elif metric_key == 'avg_success_rate':
                                ci_values.append(ci_metrics['avg_success_rate'] * 100)

                if not y_values:
                    continue

                label = f"M={n_na}"
                line, = ax.plot(
                    malicious_ratios[:len(y_values)],
                    y_values,
                    linewidth=2,
                    linestyle='-',
                    marker=markers[idx % len(markers)],
                    markersize=6,
                    label=label
                )
                legend_handles[label] = line

            if legend_handles:
                ax.legend(list(legend_handles.values()), list(legend_handles.keys()))

            if metric_key in ('avg_success_rate', 'malicious_na_selection_rate'):
                ax.set_ylim(0, 100)

            xtick_values = list(range(0, 55, 5))
            ax.set_xlabel('Malicious CA Ratio (%)')
            ax.set_xticks(xtick_values)
            ax.set_xticklabels([f"{v}" for v in xtick_values])
            ax.tick_params(axis='both', labelbottom=True)
            ax.set_xlim(0, 50)

            plt.tight_layout()
            chart_file = output_dir / f'ca_scale_sensitivity_{metric_key}.pdf'
            plt.savefig(chart_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated chart: {chart_file}")
            generated_any = True

        return generated_any

    def _plot_comparison_charts(self, strategy_data, malicious_counts, comparison_dir, n_na):
        """Plot strategy comparison charts."""

        if not strategy_data or not malicious_counts:
            print("Missing data for plotting.")
            return False

        strategy_order = [
            'random', 'balanced', 'hunger', 'weighted', 'round_robin', 'reputation',
            'adaptive', 'multi_criteria', 'psg', 'ga', 'ddqn'
        ]

        strategy_colors = {
            'random': '#1f77b4',          # blue
            'reputation': '#ff7f0e',      # orange
            'balanced': '#2ca02c',        # green
            'hunger': '#d62728',          # red
            'weighted': '#9467bd',        # purple
            'round_robin': '#8c564b',     # brown
            'adaptive': '#e377c2',        # pink
            'multi_criteria': '#bcbd22',  # olive
            'psg': '#17becf',             # cyan
            'ga': '#ff1493',              # deep pink
            'ddqn': '#000000'             # black
        }

        strategy_markers = {
            'random': 'o',          # circle
            'reputation': 's',      # square
            'balanced': '^',        # triangle_up
            'hunger': 'v',          # triangle_down
            'weighted': 'D',        # diamond
            'round_robin': 'p',     # pentagon
            'adaptive': '*',        # star
            'multi_criteria': 'h',  # hexagon
            'psg': 'X',             # x filled
            'ga': 'P',              # plus filled
            'ddqn': 'd'             # thin diamond
        }

        axis_configs = {
            'malicious_na_selection_rate': {
                'title': '',
                'ylabel': 'Malicious CA\nSelection Rate (%)'
            },
            'avg_delay_level': {
                'title': '',
                'ylabel': 'Delay Rate (%)'
            },
            'reputation_change': {
                'title': '',
                'ylabel': 'Reputation Delta'
            },
            'avg_success_rate': {
                'title': '',
                'ylabel': 'Success Rate (%)'
            }
        }

        ratio_values = [count / n_na * 100 for count in malicious_counts]
        x_values = ratio_values

        display_names = {
            'random': 'Random',
            'weighted': 'Weighted',
            'round_robin': 'Round_Robin',
            'reputation': 'Reputation',
            'psg': 'PSG',
            'ga': 'GA',
            'ddqn': 'DRL-DPKI'
        }

        generated_any = False

        # Use Times New Roman and larger fonts for comparison charts
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'font.size': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'pdf.fonttype': 42,  # embed TrueType for vector text in PDF
            'ps.fonttype': 42,
        })

        for metric_key, axis_cfg in axis_configs.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(axis_cfg['title'], fontsize=20)
            ax.set_ylabel(axis_cfg['ylabel'], fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.3)

            legend_handles = {}

            for strategy_name in strategy_order:
                if strategy_name not in strategy_data:
                    continue
                
                # Handle possibly-None entries
                result_data = strategy_data[strategy_name]
                if result_data is None:
                    continue

                metrics_list = result_data.get('metrics', [])
                if not metrics_list:
                    continue

                y_values = []
                ci_values = None

                ci_list = result_data.get('metrics_ci95')
                if isinstance(ci_list, list) and len(ci_list) == len(metrics_list):
                    ci_values = []

                for idx, metrics in enumerate(metrics_list):
                    if metrics is None:
                        y_values.append(np.nan)
                        if ci_values is not None:
                            ci_values.append(np.nan)
                        continue

                    if metric_key == 'malicious_na_selection_rate':
                        y_values.append(metrics['malicious_na_selection_rate'] * 100)
                    elif metric_key == 'avg_delay_level':
                        y_values.append(metrics['avg_delay_level'])
                    elif metric_key == 'reputation_change':
                        y_values.append(metrics['reputation_change'])
                    elif metric_key == 'avg_success_rate':
                        y_values.append(metrics['avg_success_rate'] * 100)

                    if ci_values is not None:
                        ci_metrics = ci_list[idx]
                        if ci_metrics is None:
                            ci_values.append(np.nan)
                        else:
                            if metric_key == 'malicious_na_selection_rate':
                                ci_values.append(ci_metrics['malicious_na_selection_rate'] * 100)
                            elif metric_key == 'avg_delay_level':
                                ci_values.append(ci_metrics['avg_delay_level'])
                            elif metric_key == 'reputation_change':
                                ci_values.append(ci_metrics['reputation_change'])
                            elif metric_key == 'avg_success_rate':
                                ci_values.append(ci_metrics['avg_success_rate'] * 100)

                if not y_values:
                    continue

                display_name = display_names.get(
                    strategy_name,
                    strategy_name.replace('_', ' ').title()
                )

                color = strategy_colors.get(strategy_name)

                line, = ax.plot(
                    x_values[:len(y_values)],
                    y_values,
                    marker=strategy_markers.get(strategy_name, 'o'),
                    linewidth=2,
                    label=display_name,
                    color=color
                )
                legend_handles[display_name] = line

                if ci_values is not None:
                    y_arr = np.array(y_values, dtype=float)
                    ci_arr = np.array(ci_values, dtype=float)
                    mask = np.isfinite(y_arr) & np.isfinite(ci_arr)
                    if np.any(mask):
                        x_arr = np.array(x_values[:len(y_values)], dtype=float)
                        lower = y_arr - ci_arr
                        upper = y_arr + ci_arr
                        ax.fill_between(
                            x_arr,
                            lower,
                            upper,
                            where=mask,
                            interpolate=True,
                            color=color,
                            alpha=0.15,
                            linewidth=0
                        )

            if legend_handles:
                if metric_key == 'malicious_na_selection_rate':
                    ax.legend(list(legend_handles.values()), list(legend_handles.keys()), loc='upper left', fontsize=legend_fontsize)
                else:
                    ax.legend(list(legend_handles.values()), list(legend_handles.keys()), fontsize=legend_fontsize)

            if metric_key in ('avg_success_rate', 'malicious_na_selection_rate'):
                ax.set_ylim(0, 100)

            # Set x-axis ticks at 5% intervals
            xtick_values = list(range(0, 55, 5))
            ax.set_xlabel('Malicious CA Ratio (%)', fontsize=label_fontsize)
            ax.set_xticks(xtick_values)
            ax.set_xticklabels([f"{v}" for v in xtick_values], fontsize=tick_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize, labelbottom=True)
            ax.set_xlim(0, 50)

            plt.tight_layout()
            chart_file = comparison_dir / f'comparison_charts_{metric_key}.pdf'
            plt.savefig(chart_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated chart: {chart_file}")
            generated_any = True

        return generated_any

    def _plot_drl_slidewindow_ablation_charts(self, strategy_data, malicious_counts, comparison_dir, n_na):
        if not strategy_data or not malicious_counts:
            print("Missing data for plotting.")
            return False

        strategy_order = ['drl_full', 'drl_ablation_no_slidewindow']

        display_names = {
            'drl_full': 'DRL-DPKI (Full)',
            'drl_ablation_no_slidewindow': 'DRL-DPKI (w/o temporal sliding window)'
        }

        strategy_colors = {
            'drl_full': '#000000',
            'drl_ablation_no_slidewindow': '#d62728'
        }

        strategy_markers = {
            'drl_full': 'd',
            'drl_ablation_no_slidewindow': 'o'
        }

        axis_configs = {
            'malicious_na_selection_rate': {
                'title': '',
                'ylabel': 'Malicious CA\nSelection Rate (%)'
            },
            'avg_delay_level': {
                'title': '',
                'ylabel': 'Delay Rate (%)'
            },
            'reputation_change': {
                'title': '',
                'ylabel': 'Reputation Delta'
            },
            'avg_success_rate': {
                'title': '',
                'ylabel': 'Success Rate (%)'
            }
        }

        ratio_values = [count / n_na * 100 for count in malicious_counts]
        x_values = ratio_values

        generated_any = False

        label_fontsize = 40
        tick_fontsize = 30
        title_fontsize = 30
        legend_fontsize = 24

        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'font.size': label_fontsize,
            'font.weight': 'normal',
            'axes.titlesize': title_fontsize,
            'axes.titleweight': 'normal',
            'axes.labelsize': label_fontsize,
            'axes.labelweight': 'normal',
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'legend.fontsize': legend_fontsize,
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        })

        for metric_key, axis_cfg in axis_configs.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_ylabel(axis_cfg['ylabel'], fontsize=label_fontsize, fontweight='normal')
            ax.grid(True, linestyle='--', alpha=0.3)

            legend_handles = {}

            for strategy_name in strategy_order:
                if strategy_name not in strategy_data:
                    continue

                result_data = strategy_data[strategy_name]
                if result_data is None:
                    continue

                metrics_list = result_data.get('metrics', [])
                if not metrics_list:
                    continue

                y_values = []
                ci_values = None

                ci_list = result_data.get('metrics_ci95')
                if isinstance(ci_list, list) and len(ci_list) == len(metrics_list):
                    ci_values = []

                for idx, metrics in enumerate(metrics_list):
                    if metrics is None:
                        y_values.append(np.nan)
                        if ci_values is not None:
                            ci_values.append(np.nan)
                        continue

                    if metric_key == 'malicious_na_selection_rate':
                        y_values.append(metrics['malicious_na_selection_rate'] * 100)
                    elif metric_key == 'avg_delay_level':
                        y_values.append(metrics['avg_delay_level'])
                    elif metric_key == 'reputation_change':
                        y_values.append(metrics['reputation_change'])
                    elif metric_key == 'avg_success_rate':
                        y_values.append(metrics['avg_success_rate'] * 100)

                    if ci_values is not None:
                        ci_metrics = ci_list[idx]
                        if ci_metrics is None:
                            ci_values.append(np.nan)
                        else:
                            if metric_key == 'malicious_na_selection_rate':
                                ci_values.append(ci_metrics['malicious_na_selection_rate'] * 100)
                            elif metric_key == 'avg_delay_level':
                                ci_values.append(ci_metrics['avg_delay_level'])
                            elif metric_key == 'reputation_change':
                                ci_values.append(ci_metrics['reputation_change'])
                            elif metric_key == 'avg_success_rate':
                                ci_values.append(ci_metrics['avg_success_rate'] * 100)

                if not y_values:
                    continue

                display_name = display_names.get(strategy_name, strategy_name.replace('_', ' ').title())
                color = strategy_colors.get(strategy_name)

                line, = ax.plot(
                    x_values[:len(y_values)],
                    y_values,
                    marker=strategy_markers.get(strategy_name, 'o'),
                    linewidth=2,
                    label=display_name,
                    color=color
                )
                legend_handles[display_name] = line

                if ci_values is not None:
                    y_arr = np.array(y_values, dtype=float)
                    ci_arr = np.array(ci_values, dtype=float)
                    mask = np.isfinite(y_arr) & np.isfinite(ci_arr)
                    if np.any(mask):
                        x_arr = np.array(x_values[:len(y_values)], dtype=float)
                        lower = y_arr - ci_arr
                        upper = y_arr + ci_arr
                        ax.fill_between(
                            x_arr,
                            lower,
                            upper,
                            where=mask,
                            interpolate=True,
                            color=color,
                            alpha=0.15,
                            linewidth=0
                        )

            if legend_handles:
                if metric_key == 'malicious_na_selection_rate':
                    ax.legend(list(legend_handles.values()), list(legend_handles.keys()), loc='upper left', fontsize=legend_fontsize)
                else:
                    ax.legend(list(legend_handles.values()), list(legend_handles.keys()), fontsize=legend_fontsize)

            if metric_key in ('avg_success_rate', 'malicious_na_selection_rate'):
                ax.set_ylim(0, 100)

            xtick_values = list(range(0, 55, 5))
            ax.set_xlabel('Malicious CA Ratio (%)', fontsize=label_fontsize + 4, fontweight='normal')
            ax.set_xticks(xtick_values)
            ax.set_xticklabels([f"{v}" for v in xtick_values], fontsize=tick_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize, labelbottom=True)
            ax.set_xlim(0, 50)

            plt.tight_layout()
            chart_file = comparison_dir / f'slidewindow_ablation_{metric_key}.pdf'
            plt.savefig(chart_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated chart: {chart_file}")
            generated_any = True

        if not generated_any:
            print(
                "Warning: No slide-window ablation comparison data found "
                "(requires strategies to include drl_full / drl_ablation_no_slidewindow)"
            )

        return generated_any

    def _plot_comparison_table_from_data(self, comparison_dir):
        """Extract data and plot a comparison table."""
        strategy_data = self.data.get('strategies', {})
        if not strategy_data:
            return
            
        final_table_data = []
        for strategy_name, data in strategy_data.items():
            if not data or not data.get('metrics'):
                continue
            final_metrics = data['metrics'][-1]
            if final_metrics is None:
                continue
            final_table_data.append({
                'Strategy Name': strategy_name.capitalize(),
                'Malicious NA Selections': final_metrics['malicious_na_selection_count'],
                'Malicious NA Rate': f"{final_metrics['malicious_na_selection_rate']:.1%}",
                'Average Delay Level': f"{final_metrics['avg_delay_level']:.3f}",
                'Reputation Change': f"{final_metrics['reputation_change']:+.1f}",
                'Average Success Rate': f"{final_metrics['avg_success_rate']:.1%}"
            })

        if final_table_data:
            self._plot_comparison_table(final_table_data, comparison_dir, self.n_time_points, self.n_na, self.seed)

    def _plot_comparison_table(self, table_data, comparison_dir, n_time_points, n_na, seed):
        """Plot a comparison table image."""
        if not table_data:
            print("No data available to generate a comparison table.")
            return False

        # Create the table figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Strategy Name', 'Malicious NA Selections', 'Malicious NA Rate', 'Average Delay Level', 'Reputation Change', 'Average Success Rate']
        cell_data = []
        
        for row in table_data:
            cell_data.append([
                row['Strategy Name'],
                str(row['Malicious NA Selections']),
                row['Malicious NA Rate'],
                row['Average Delay Level'],
                row['Reputation Change'],
                row['Average Success Rate']
            ])
        
        # Create the table
        table = ax.table(cellText=cell_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1.2, 2)
        
        # Colors
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(cell_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                # Special coloring
                if j == 4:  # Reputation Change column
                    change_str = cell_data[i-1][j]
                    if '+' in change_str:
                        table[(i, j)].set_facecolor('#E8F5E8')
                    elif '-' in change_str:
                        table[(i, j)].set_facecolor('#FFE8E8')
        
        # Title
        title = f'Strategy Comparison Summary\n'
        title += f'NA Count: {n_na}, Time Points: {n_time_points}, Seed: {seed}\n'
        title += f'Test Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        plt.title(title, fontsize=20, fontweight='bold', pad=20)
        
        table_file = comparison_dir / 'comparison_table.png'
        plt.savefig(table_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated table: {table_file}")
        return True

    def _plot_drl_cold_warm_charts(self, results, malicious_counts, comparison_dir, n_na):
        """Plot DRL-DPKI cold/warm comparison charts."""
        
        if not results or not malicious_counts:
             print("Error: Missing DRL cold/warm plotting data.")
             return

        axis_configs = {
            'malicious_na_selection_rate': {
                'title': '',
                'ylabel': 'Malicious CA Selection Rate (%)'
            },
            'avg_delay_level': {
                'title': '',
                'ylabel': 'Delay Rate (%)'
            },
            'reputation_change': {
                'title': '',
                'ylabel': 'Reputation Delta'
            },
            'avg_success_rate': {
                'title': '',
                'ylabel': 'Success Rate (%)'
            }
        }

        ratio_values = [count / n_na * 100 for count in malicious_counts]
        x_values = ratio_values

        display_names = {
            'cold': 'Cold Start',
            'warm': 'Warm Start'
        }

        colors = {
            'cold': '#1f77b4',  # blue
            'warm': '#ff7f0e'   # orange
        }

        markers = {
            'cold': 'o',  # circle
            'warm': '^'   # triangle_up
        }

        # Times New Roman and vector fonts
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'font.size': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        for metric_key, axis_cfg in axis_configs.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(axis_cfg['title'], fontsize=20)
            ax.set_ylabel(axis_cfg['ylabel'], fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.3)

            legend_handles = {}

            for mode_key in ['cold', 'warm']:
                metrics_list = results.get(mode_key, {}).get('metrics', [])
                if not metrics_list:
                    continue

                y_values = []
                for metrics in metrics_list:
                    if metrics is None:
                        y_values.append(np.nan)
                    else:
                        if metric_key == 'malicious_na_selection_rate':
                            y_values.append(metrics['malicious_na_selection_rate'] * 100)
                        elif metric_key == 'avg_delay_level':
                            y_values.append(metrics['avg_delay_level'])
                        elif metric_key == 'reputation_change':
                            y_values.append(metrics['reputation_change'])
                        elif metric_key == 'avg_success_rate':
                            y_values.append(metrics['avg_success_rate'] * 100)

                if not y_values:
                    continue

                line, = ax.plot(
                    x_values[:len(y_values)],
                    y_values,
                    marker=markers.get(mode_key, 'o'),
                    linewidth=2,
                    label=display_names.get(mode_key, mode_key.title()),
                    color=colors.get(mode_key)
                )
                legend_handles[mode_key] = line

            if legend_handles:
                ax.legend(list(legend_handles.values()), [display_names[k] for k in legend_handles.keys()], fontsize=16)

            if metric_key in ('avg_success_rate', 'malicious_na_selection_rate'):
                ax.set_ylim(0, 100)

            # Set x-axis ticks at 5% intervals
            xtick_values = list(range(0, 55, 5))
            ax.set_xlabel('Malicious CA Ratio (%)', fontsize=18)
            ax.set_xticks(xtick_values)
            ax.set_xticklabels([f"{v}" for v in xtick_values], fontsize=16)
            ax.tick_params(axis='both', labelsize=16, labelbottom=True)
            ax.set_xlim(0, 50)

            plt.tight_layout()
            chart_file = comparison_dir / f'drl_cold_warm_{metric_key}.pdf'
            plt.savefig(chart_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated chart: {chart_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('--file', type=str, required=False, help='Path to the results JSON file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save plots (default: same as JSON file directory)')
    
    args = parser.parse_args()

    result_file = args.file
    if result_file is None:
        # Try to auto-find the most recent comparison_results.json
        default_dir = Path(__file__).parent / 'strategy_comparison'
        possible_files = sorted(default_dir.rglob('*_results.json'), key=os.path.getmtime, reverse=True)
        if possible_files:
            result_file = str(possible_files[0])
            print(f"No file specified; using the latest results file: {result_file}")
        else:
            print("Error: No file specified and no default results file found.")
            parser.print_help()
            sys.exit(1)
    
    if not os.path.exists(result_file):
        print(f"Error: File not found: {result_file}")
        sys.exit(1)
        
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(result_file))
        
    plotter = ResultPlotter(result_file)
    plotter.plot_all(output_dir)

if __name__ == "__main__":
    main()
