#!/usr/bin/env python3
"""
GPU monitoring utility for Jetson.

Shows real-time GPU utilization, memory usage, and power consumption.
"""

import subprocess
import time
import sys
import re
from typing import Dict, Optional


def parse_tegrastats(line: str) -> Dict[str, str]:
    """Parse a line of tegrastats output."""
    stats = {}
    
    # RAM usage: "RAM 3743/7620MB"
    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    if ram_match:
        used, total = ram_match.groups()
        stats['ram_used_mb'] = used
        stats['ram_total_mb'] = total
        stats['ram_pct'] = f"{(int(used) / int(total)) * 100:.1f}"
    
    # GPU frequency: "GR3D_FREQ 45%"
    gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
    if gpu_match:
        stats['gpu_util'] = gpu_match.group(1)
    
    # GPU temp: "gpu@46.781C"
    temp_match = re.search(r'gpu@([\d.]+)C', line)
    if temp_match:
        stats['gpu_temp'] = temp_match.group(1)
    
    # Power: "VDD_IN 4888mW/4888mW"
    power_match = re.search(r'VDD_IN (\d+)mW/(\d+)mW', line)
    if power_match:
        instant, avg = power_match.groups()
        stats['power_instant_mw'] = instant
        stats['power_avg_mw'] = avg
    
    # CPU+GPU power: "VDD_CPU_GPU_CV 555mW/555mW"
    cpu_gpu_power = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
    if cpu_gpu_power:
        stats['cpu_gpu_power_mw'] = cpu_gpu_power.group(1)
    
    return stats


def format_stats(stats: Dict[str, str]) -> str:
    """Format stats for display."""
    if not stats:
        return "Waiting for data..."
    
    ram_used = int(stats.get('ram_used_mb', 0))
    ram_total = int(stats.get('ram_total_mb', 1))
    ram_pct = float(stats.get('ram_pct', 0))
    
    gpu_util = int(stats.get('gpu_util', 0))
    gpu_temp = float(stats.get('gpu_temp', 0))
    
    power_w = int(stats.get('power_avg_mw', 0)) / 1000
    cpu_gpu_w = int(stats.get('cpu_gpu_power_mw', 0)) / 1000
    
    # Create progress bar for GPU
    bar_width = 20
    filled = int(bar_width * gpu_util / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    # Create progress bar for RAM
    ram_filled = int(bar_width * ram_pct / 100)
    ram_bar = '█' * ram_filled + '░' * (bar_width - ram_filled)
    
    return (
        f"GPU: {bar} {gpu_util:3d}%  "
        f"Temp: {gpu_temp:5.1f}°C  "
        f"RAM: {ram_bar} {ram_used:4d}/{ram_total}MB ({ram_pct:4.1f}%)  "
        f"Power: {power_w:5.1f}W (CPU+GPU: {cpu_gpu_w:4.1f}W)"
    )


def monitor_gpu(interval_ms: int = 1000, duration_sec: Optional[int] = None):
    """
    Monitor GPU in real-time.
    
    Args:
        interval_ms: Update interval in milliseconds
        duration_sec: How long to run (None = forever)
    """
    try:
        # Start tegrastats
        proc = subprocess.Popen(
            ['tegrastats', '--interval', str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print(f"Monitoring GPU (Ctrl+C to stop)...")
        print()
        
        start_time = time.time()
        
        for line in proc.stdout:
            stats = parse_tegrastats(line.strip())
            
            # Clear line and print stats
            sys.stdout.write('\r' + ' ' * 150 + '\r')
            sys.stdout.write(format_stats(stats))
            sys.stdout.flush()
            
            # Check duration
            if duration_sec and (time.time() - start_time) >= duration_sec:
                break
        
        proc.terminate()
        proc.wait()
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\nStopped.")
        proc.terminate()
        proc.wait()
    except FileNotFoundError:
        print("Error: tegrastats not found. This tool requires Jetson Linux.")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Jetson GPU usage')
    parser.add_argument('-i', '--interval', type=int, default=500,
                       help='Update interval in milliseconds (default: 500)')
    parser.add_argument('-d', '--duration', type=int, default=None,
                       help='Duration in seconds (default: run forever)')
    
    args = parser.parse_args()
    
    monitor_gpu(args.interval, args.duration)
