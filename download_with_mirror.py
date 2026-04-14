#!/usr/bin/env python3
"""
Robust Qwen2.5-1.5B model downloader with HuggingFace mirror support
Supports automatic failover and resume capability
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAVE_DIR = str(Path.home() / "models" / "Qwen--Qwen2.5-1.5B-Instruct")
PROGRESS_FILE = str(Path.home() / "models" / ".download_progress.json")
LOG_FILE = str(Path.home() / "models" / "download.log")

# HuggingFace mirror endpoints (China-friendly)
HF_MIRRORS = [
    "https://hf-mirror.com",  # Popular China mirror
    "https://huggingface.co",  # Official (may be slow in China)
]

class DownloadMonitor:
    def __init__(self, save_dir, progress_file):
        self.save_dir = Path(save_dir)
        self.progress_file = progress_file
        self.start_time = time.time()
        self.last_size = 0
        self.last_check_time = time.time()
        
    def get_dir_size(self):
        """Calculate total size of download directory"""
        if not self.save_dir.exists():
            return 0
        total = 0
        for path in self.save_dir.rglob('*'):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except:
                    pass
        return total
    
    def save_progress(self, status, method, error=None):
        """Save download progress to file"""
        current_size = self.get_dir_size()
        elapsed = time.time() - self.start_time
        speed = (current_size - self.last_size) / (time.time() - self.last_check_time) if time.time() > self.last_check_time else 0
        
        progress = {
            'status': status,
            'method': method,
            'downloaded_bytes': current_size,
            'downloaded_gb': round(current_size / (1024**3), 2),
            'elapsed_seconds': int(elapsed),
            'speed_mbps': round(speed / (1024**2), 2),
            'timestamp': datetime.now().isoformat(),
            'error': str(error) if error else None
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        self.last_size = current_size
        self.last_check_time = time.time()
        return progress
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(LOG_FILE, 'a') as f:
            f.write(log_msg + '\n')

def download_with_hf(monitor, mirror_url=None):
    """Download using HuggingFace Hub"""
    monitor.log(f"🚀 Starting download via HuggingFace (mirror: {mirror_url or 'default'})")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Set mirror endpoint if provided
        if mirror_url:
            os.environ['HF_ENDPOINT'] = mirror_url
        
        monitor.save_progress('downloading', 'huggingface', None)
        
        # Download with resume capability
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=SAVE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint=mirror_url
        )
        
        monitor.log("✅ HuggingFace download completed successfully")
        monitor.save_progress('completed', 'huggingface', None)
        return True
        
    except KeyboardInterrupt:
        monitor.log("⚠️  Download interrupted by user")
        monitor.save_progress('interrupted', 'huggingface', 'User interrupted')
        raise
    except Exception as e:
        monitor.log(f"❌ HuggingFace download failed: {e}")
        monitor.save_progress('failed', 'huggingface', str(e))
        return False

def download_with_retry(monitor, max_retries=3):
    """Download with automatic retry and failover"""
    
    # Try HuggingFace mirrors
    for mirror in HF_MIRRORS:
        monitor.log(f"🔄 Attempting download with HF mirror: {mirror}")
        for attempt in range(max_retries):
            monitor.log(f"   Attempt {attempt + 1}/{max_retries}")
            try:
                if download_with_hf(monitor, mirror):
                    return True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                monitor.log(f"   Retry {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    monitor.log(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        monitor.log(f"❌ All attempts failed for mirror: {mirror}")
    
    return False

def main():
    """Main download function"""
    print("="*70)
    print("🤖 Qwen2.5-1.5B-Instruct Model Downloader")
    print("="*70)
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize monitor
    monitor = DownloadMonitor(SAVE_DIR, PROGRESS_FILE)
    
    # Check existing files
    existing_size = monitor.get_dir_size()
    if existing_size > 0:
        monitor.log(f"📂 Found existing download: {existing_size / (1024**3):.2f} GB")
        monitor.log("📥 Will resume download...")
    
    # Start download with retry
    try:
        success = download_with_retry(monitor)
        
        if success:
            final_size = monitor.get_dir_size()
            elapsed = time.time() - monitor.start_time
            avg_speed = final_size / elapsed / (1024**2) if elapsed > 0 else 0
            
            print("\n" + "=" * 70)
            print("✅ DOWNLOAD COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"📁 Location: {SAVE_DIR}")
            print(f"💾 Total size: {final_size / (1024**3):.2f} GB")
            print(f"⏱️  Time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print(f"🚀 Average speed: {avg_speed:.2f} MB/s")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("❌ DOWNLOAD FAILED")
            print("=" * 70)
            print(f"📂 Partial download saved to: {SAVE_DIR}")
            print(f"💾 Downloaded: {monitor.get_dir_size() / (1024**3):.2f} GB")
            print("🔄 You can re-run this script to resume")
            print("=" * 70)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print(f"📂 Partial download saved to: {SAVE_DIR}")
        print(f"💾 Downloaded: {monitor.get_dir_size() / (1024**3):.2f} GB")
        print("🔄 Run this script again to resume")
        sys.exit(130)

if __name__ == "__main__":
    main()
