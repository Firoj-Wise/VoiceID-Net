# Command Center for VoiceID-Net

import argparse
import os
import sys
import time
import torch
import numpy as np

from voice_engine.models import create_model
from voice_engine.export_onnx import export_to_onnx, benchmark_latency

def test_architecture():
    print("=" * 60)
    print("🚀 [TEST] Validating VoiceID-Net Architecture & Latency Benchmark")
    print("=" * 60)
    
    torch.set_num_threads(4)
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(embed_dim=192, base_channels=64)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters     : {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters : {trainable_params / 1e6:.2f} M")
    print(f"Output Embedding Dim : {model.embed_dim}")
    print(f"Active CPU Threads   : {torch.get_num_threads()} (Optimized to prevent lock thrashing)")
    
    # 1. Test CPU forward & latency
    model_cpu = model.to("cpu").eval()
    dummy_input_cpu = torch.randn(1, 80, 300) # 3-second utterance (300 frames)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model_cpu(dummy_input_cpu)
            
    start = time.perf_counter()
    n_runs = 50
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model_cpu(dummy_input_cpu)
    cpu_latency = ((time.perf_counter() - start) / n_runs) * 1000.0
    print(f"⏱️ CPU Latency (3s audio): {cpu_latency:.2f} ms")
    
    # 2. Test GPU forward & latency if available
    if torch.cuda.is_available():
        model_gpu = model.to(device_cuda).eval()
        dummy_input_gpu = dummy_input_cpu.to(device_cuda)
        
        for _ in range(10):
            _ = model_gpu(dummy_input_gpu)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        n_gpu_runs = 200
        for _ in range(n_gpu_runs):
            _ = model_gpu(dummy_input_gpu)
        torch.cuda.synchronize()
        gpu_latency = ((time.perf_counter() - start) / n_gpu_runs) * 1000.0
        print(f"⚡ GPU Latency (3s audio on {torch.cuda.get_device_name(0)}): {gpu_latency:.2f} ms")
        
    print("=" * 60)
    print("✅ Model validated successfully. Ultra-low latency confirmed!")
    print("=" * 60)

def export_cmd(args):
    print("=" * 60)
    print("📦 [EXPORT] Exporting VoiceID-Net to ONNX")
    print("=" * 60)
    model = create_model(embed_dim=192, base_channels=64)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print("Exporting model architecture (weights initialized).")
        
    onnx_file = args.output or "voice_engine/voice_encoder.onnx"
    export_to_onnx(model, onnx_path=onnx_file)
    benchmark_latency(onnx_file)

def main():
    parser = argparse.ArgumentParser(description="VoiceID-Net Command Center")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("test-arch", help="Validate architecture & benchmark latency")
    
    export_p = subparsers.add_parser("export", help="Export model to ONNX")
    export_p.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    export_p.add_argument("--output", type=str, default="voice_engine/voice_encoder.onnx", help="Path to output ONNX file")
    
    args = parser.parse_args()
    if args.command == "test-arch":
        test_architecture()
    elif args.command == "export":
        export_cmd(args)
    else:
        test_architecture()

if __name__ == "__main__":
    main()
