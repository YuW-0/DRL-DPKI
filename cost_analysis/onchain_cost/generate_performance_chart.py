#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NA performance test data visualization script.
Generates a performance analysis chart.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def load_performance_data(json_file):
    """Load performance data from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_chinese_font():
    """Set up CJK-capable fonts (best effort)."""
    # Try common system fonts
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue

def create_performance_chart(data):
    """Create the performance chart."""
    # Configure fonts
    setup_chinese_font()
    
    # Extract data
    rounds = data['rounds']
    round_numbers = data['roundNumbers']
    avg_na_times = data['avgNATimes']
    
    # Create chart
    plt.figure(figsize=(12, 8))
    
    # Plot per-round average NA execution time
    plt.plot(round_numbers, avg_na_times, 
             marker='s', linewidth=1.8, markersize=6,
             label='Per-round average NA execution time', color='#2ca02c')
    
    # Add a horizontal baseline for the overall average
    avg_na_execution_time = data['averages']['naTime']
    plt.axhline(y=avg_na_execution_time, color='#2ca02c', linestyle='--', linewidth=2,
                label=f'Overall average: {avg_na_execution_time:.2f}ms', alpha=0.8)
    
    # Chart styling
    plt.xlabel('Round', fontsize=12, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.title(f'NA Performance Test Results - {rounds} Rounds', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Axes range
    plt.xlim(0.5, rounds + 0.5)
    plt.ylim(0, max(avg_na_times) * 1.1)
    
    # X ticks
    plt.xticks(range(1, rounds + 1))
    
    # Stats box
    avg_data = data['averages']
    stats_text = f"""Summary:
Per-NA exec time: {avg_data['naTime']:.2f}ms
NA count: {data['naCount']}
Rounds: {rounds}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Layout
    plt.tight_layout()
    
    return plt

def main():
    """Main entry point."""
    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Data file path
    json_file = os.path.join(script_dir, 'performance-data.json')
    
    # Ensure the data file exists
    if not os.path.exists(json_file):
        print(f"Error: data file not found: {json_file}")
        print("Run na-batch-rotation.js first to generate the data file.")
        return
    
    try:
        # Load data
        print("Loading performance data...")
        data = load_performance_data(json_file)
        
        # Create chart
        print("Generating performance chart...")
        plt = create_performance_chart(data)
        
        # Save image
        output_file = os.path.join(script_dir, 'na_performance_chart.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Performance chart generated: {output_file}")
        print(f"Chart contains {data['rounds']} rounds of test data.")
        print("Shows NA execution time curve and overall average baseline.")
        
        # Show chart (optional)
        # plt.show()
        
    except Exception as e:
        print(f"Error while generating chart: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
