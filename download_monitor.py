#!/usr/bin/env python3
"""
Real-time download monitor for Qwen2.5-1.5B model
Displays download progress, speed, and file size changes
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

SAVE_DIR = str(Path.home() / "models" / "Qwen--Qwen2.5-1.5B-Instruct")
PROGRESS_FILE = str(Path.home() / "models" / ".download_progress.json")

def get_dir_size(directory):
    """Calculate total size of directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except:
                    pass
    except:
        pass
    return total

def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def format_time(seconds):
    """Format seconds to human readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def estimate_remaining(current_size, total_estimate, speed):
    """Estimate remaining time"""
    if speed <= 0:
        return "Unknown"
    remaining_bytes = total_estimate - current_size
    if remaining_bytes <= 0:
        return "Almost done"
    remaining_seconds = remaining_bytes / speed
    return format_time(remaining_seconds)

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def read_progress_file():
    """Read progress from JSON file"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return None

def count_files(directory):
    """Count files in directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob('*') if _.is_file())

def monitor_download(refresh_interval=3):
    """Monitor download progress in real-time"""
    
    print("🔍 Starting download monitor...")
    print(f"📂 Monitoring: {SAVE_DIR}")
    print(f"🔄 Refresh interval: {refresh_interval}s")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    last_size = get_dir_size(SAVE_DIR)
    last_check = time.time()
    
    # Estimated total size for Qwen2.5-1.5B (approximate)
    ESTIMATED_TOTAL = 3 * 1024 * 1024 * 1024  # ~3 GB
    
    try:
        while True:
            time.sleep(refresh_interval)
            
            current_time = time.time()
            current_size = get_dir_size(SAVE_DIR)
            elapsed = current_time - start_time
            interval = current_time - last_check
            
            # Calculate speed
            size_diff = current_size - last_size
            instant_speed = size_diff / interval if interval > 0 else 0
            avg_speed = current_size / elapsed if elapsed > 0 else 0
            
            # Read progress file
            progress_data = read_progress_file()
            
            # Clear screen and display
            clear_screen()
            
            print("=" * 70)
            print("🤖 Qwen2.5-1.5B-Instruct Download Monitor")
            print("=" * 70)
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Elapsed: {format_time(elapsed)}")
            print("-" * 70)
            
            # Size information
            print(f"💾 Downloaded: {format_bytes(current_size)} / ~{format_bytes(ESTIMATED_TOTAL)}")
            progress_pct = (current_size / ESTIMATED_TOTAL * 100) if ESTIMATED_TOTAL > 0 else 0
            progress_bar_length = 50
            filled = int(progress_bar_length * min(progress_pct, 100) / 100)
            bar = '█' * filled + '░' * (progress_bar_length - filled)
            print(f"📊 Progress: [{bar}] {min(progress_pct, 100):.1f}%")
            
            print("-" * 70)
            
            # Speed information
            print(f"🚀 Instant speed: {format_bytes(instant_speed)}/s")
            print(f"📈 Average speed: {format_bytes(avg_speed)}/s")
            
            # Estimate remaining time
            if instant_speed > 0:
                eta = estimate_remaining(current_size, ESTIMATED_TOTAL, instant_speed)
                print(f"⏳ ETA: {eta}")
            
            print("-" * 70)
            
            # Progress file data
            if progress_data:
                print(f"📝 Status: {progress_data.get('status', 'unknown')}")
                print(f"🔧 Method: {progress_data.get('method', 'unknown')}")
                if progress_data.get('error'):
                    print(f"⚠️  Error: {progress_data['error']}")
            
            print("-" * 70)
            
            # File count
            file_count = count_files(SAVE_DIR)
            print(f"📄 Files: {file_count}")
            
            # Stall detection
            if size_diff == 0 and elapsed > 30:
                print("\n⚠️  WARNING: No progress detected! Download may be stalled.")
                print("   Check the download script or network connection.")
            
            print("=" * 70)
            print(f"🔄 Next update in {refresh_interval}s... (Press Ctrl+C to stop)")
            
            # Update for next iteration
            last_size = current_size
            last_check = current_time
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped by user")
        print(f"📊 Final size: {format_bytes(current_size)}")
        print(f"⏱️  Total time: {format_time(elapsed)}")

def main():
    """Main function"""
    default_interval = 3
    
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
            monitor_download(interval)
        except ValueError:
            print("Usage: python download_monitor.py [refresh_interval_seconds]")
            print("Example: python download_monitor.py 3")
            sys.exit(1)
    else:
        monitor_download(default_interval)

if __name__ == "__main__":
    main()
