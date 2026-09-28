# VoiceID-Net: Ultra-Low Latency Speaker Verification Engine

VoiceID-Net is a high-speed, lightweight State-of-the-Art (SOTA) Speaker Verification and Recognition Engine designed for high-throughput production deployment.

---

## Performance Benchmarks

| Device | Latency (3.0s Speech Utterance) | Throughput |
| :--- | :--- | :--- |
| **NVIDIA GeForce RTX 4090 (GPU)** | **4.56 ms** | ~650+ utterances / sec (batched) |
| **Intel / AMD CPU (4 Threads)** | **12.43 ms** | ~80+ utterances / sec |
| **ONNX Runtime (CPU)** | **3.46 ms - 4.68 ms** | Standalone deployment (zero PyTorch dependency) |

- **Parameters:** 2.65 M (compact, memory-efficient)
- **Embedding Dimension:** 192 (unit-sphere normalized)
- **Acoustic Frontend:** 80-channel log-Mel filterbank (25ms window, 10ms frame shift, 16kHz) + Cepstral Mean Normalization (CMN)
- **Pooling Mechanism:** Context-Aware Attentive Statistics Pooling (ASP)

---

## Quick Start

### 1. Benchmark & Validate Architecture
```bash
python -m voice_engine.main test-arch
```

### 2. Export Model to ONNX
```bash
python -m voice_engine.main export --output voice_engine/voice_encoder.onnx
```

### 3. Verification & Cosine Similarity
```python
from voice_engine.infer import VoiceVerifier

verifier = VoiceVerifier("voice_engine/voice_encoder.onnx")

# Extract 192-dim speaker embedding
embedding = verifier.get_embedding("audio1.wav")

# Compare two audio files
similarity, is_same = verifier.verify("audio1.wav", "audio2.wav", threshold=0.60)
print(f"Cosine Similarity: {similarity:.4f} | Same Speaker: {is_same}")
```

---

## Autonomous Training Pipeline

```python
from voice_engine.trainer import RepresentationTrainer
from voice_engine.dataset import SpeechDataset, collate_speech_batches
from torch.utils.data import DataLoader

trainer = RepresentationTrainer(
    embed_dim=192,
    base_channels=64,
    lr=1e-3,
    device="cuda"
)

dataset = SpeechDataset(audio_files=["path/to/wav1.wav", "path/to/wav2.wav"])
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_speech_batches)

for epoch in range(1, 21):
    metrics = trainer.train_epoch(loader, epoch=epoch, total_epochs=20)
    trainer.save_checkpoint(f"checkpoints/voiceid_epoch_{epoch}.pt")
```
