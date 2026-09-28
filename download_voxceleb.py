#!/usr/bin/env python3
# Automated Robust Resumable VoxCeleb1 & VoxCeleb2 Dataset Downloader

import argparse
import os
import subprocess
import sys
import time

HF_BASE_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main"

# Exact expected byte sizes for VoxCeleb2 parts
VOX2_PART_SIZES = {
    "vox2_dev_aac_partaa": 10737418240,
    "vox2_dev_aac_partab": 10737418240,
    "vox2_dev_aac_partac": 10737418240,
    "vox2_dev_aac_partad": 10737418240,
    "vox2_dev_aac_partae": 10737418240,
    "vox2_dev_aac_partaf": 10737418240,
    "vox2_dev_aac_partag": 10737418240,
    "vox2_dev_aac_partah": 2315355528,
}

def download_file_with_resume(url, output_path, expected_size=None, max_retries=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filename = os.path.basename(output_path)
    
    # Check if already complete
    if os.path.exists(output_path):
        curr_size = os.path.getsize(output_path)
        if expected_size and curr_size == expected_size:
            print(f"[Verified Complete] {filename} ({curr_size / (1024**3):.2f} GB)")
            return output_path
        elif expected_size and curr_size > expected_size:
            print(f"[Warning] {filename} is larger than expected ({curr_size} vs {expected_size}), removing corrupted file...")
            os.remove(output_path)
        else:
            print(f"[Resuming] {filename} has {curr_size / (1024**3):.2f} GB of {expected_size / (1024**3) if expected_size else 'total'} GB. Continuing...")
            
    retries = 0
    while retries < max_retries:
        print(f"[Downloader] Downloading {filename} (Attempt {retries + 1}/{max_retries})...")
        cmd = ["wget", "-c", "--tries=10", "--timeout=30", "--show-progress", url, "-O", output_path]
        res = subprocess.run(cmd)
        
        if res.returncode == 0:
            if expected_size:
                curr_size = os.path.getsize(output_path)
                if curr_size == expected_size:
                    print(f"[Verified Complete] {filename} successfully downloaded!")
                    return output_path
                else:
                    print(f"[Retry] Incomplete size ({curr_size} != {expected_size}), retrying resume in 3s...")
            else:
                return output_path
                
        retries += 1
        time.sleep(3)
        
    raise RuntimeError(f"Failed to complete download for {url} after {max_retries} attempts.")

def download_vox2_dev(data_dir):
    """
    Downloads the EXACT VoxCeleb2 dataset used by the original paper authors:
    ~1,092,009 utterances from 5,994 unique speakers (~72.3 GB total).
    """
    parts = ["partaa", "partab", "partac", "partad", "partae", "partaf", "partag", "partah"]
    part_files = []
    
    download_dir = os.path.join(data_dir, "downloads")
    os.makedirs(download_dir, exist_ok=True)
    
    # 1. Download/Resume all parts
    for p in parts:
        p_name = f"vox2_dev_aac_{p}"
        url = f"{HF_BASE_URL}/vox2/{p_name}"
        dest_p = os.path.join(download_dir, p_name)
        expected_bytes = VOX2_PART_SIZES.get(p_name)
        download_file_with_resume(url, dest_p, expected_size=expected_bytes)
        part_files.append(dest_p)
        
    # 2. Concatenate parts into single zip
    full_zip = os.path.join(download_dir, "vox2_dev_aac.zip")
    if not os.path.exists(full_zip):
        print(f"[Concatenating] Merging 8 parts into {full_zip}...")
        with open(full_zip, "wb") as outfile:
            for pf in part_files:
                print(f" -> Appending {os.path.basename(pf)}...")
                with open(pf, "rb") as infile:
                    while chunk := infile.read(1024 * 1024 * 64): # 64MB buffer
                        outfile.write(chunk)
                        
        print("[Disk Management] Removing part files to preserve disk space...")
        for pf in part_files:
            if os.path.exists(pf):
                os.remove(pf)
                
    # 3. Unzip
    extract_dir = os.path.join(data_dir, "vox2_dev")
    if not os.path.exists(extract_dir):
        print(f"[Extractor] Unzipping {full_zip} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        res = subprocess.run(["unzip", "-q", full_zip, "-d", extract_dir])
        if res.returncode == 0:
            print("[Disk Management] Extraction successful. Removing zip archive to keep disk usage under 75GB...")
            os.remove(full_zip)
            
    print(f"[Setup] Exact VoxCeleb2 Dataset ready at: {extract_dir}")
    return extract_dir

def download_test_set(data_dir):
    test_zip_url = f"{HF_BASE_URL}/vox1/vox1_test_wav.zip"
    dest_zip = os.path.join(data_dir, "downloads", "vox1_test_wav.zip")
    download_file_with_resume(test_zip_url, dest_zip)
    
    extract_dir = os.path.join(data_dir, "vox1_test")
    if not os.path.exists(extract_dir):
        print(f"[Extractor] Unzipping {dest_zip} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        subprocess.run(["unzip", "-q", dest_zip, "-d", extract_dir])
    print(f"[Setup] VoxCeleb1 Test Set ready at: {extract_dir}")
    return extract_dir

def main():
    parser = argparse.ArgumentParser(description="VoxCeleb Dataset Downloader with Auto-Resume")
    parser.add_argument("--dataset", type=str, choices=["test", "vox2"], default="vox2",
                        help="'test' (1GB eval set) or 'vox2' (Exact 72.3GB paper dataset, 1.09M utterances)")
    parser.add_argument("--data-dir", type=str, default="/home/oem/wiseyak_backup/firojpaudel/voxceleb_data",
                        help="Target directory to store dataset")
    args = parser.parse_args()
    
    print("=" * 65)
    print("VoxCeleb Large-Scale Dataset Downloader (Auto-Resume Enabled)")
    print(f"Target Selection : {args.dataset}")
    print(f"Destination Path : {args.data_dir}")
    print("=" * 65)
    
    if args.dataset == "test":
        download_test_set(args.data_dir)
    elif args.dataset == "vox2":
        download_vox2_dev(args.data_dir)
        
    print("=" * 65)
    print("Download and setup complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
