# ONNX Exporter & Benchmark Engine

import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from voice_engine.models import create_model

def export_to_onnx(
    model,
    onnx_path="voice_engine/voice_encoder.onnx",
    device="cpu"
):
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 80, 300, device=device)
    
    print(f"[Export] Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch_size", 2: "num_frames"},
            "embedding": {0: "batch_size"}
        }
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[Export] ONNX model verified successfully!")
    
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()
        
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"fbank": dummy_input.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"[Export] Max absolute numerical difference: {max_diff:.2e}")
    assert max_diff < 1e-4, "Numerical divergence exceeds tolerance!"
    
    return onnx_path


def benchmark_latency(onnx_path, runs=100):
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 80, 300).astype(np.float32)
    
    for _ in range(10):
        _ = ort_session.run(None, {"fbank": dummy_input})
        
    start_time = time.perf_counter()
    for _ in range(runs):
        _ = ort_session.run(None, {"fbank": dummy_input})
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / runs) * 1000.0
    print(f"[Benchmark] CPU Inference Latency (3.0s audio): {avg_latency_ms:.2f} ms per utterance")
    return avg_latency_ms
