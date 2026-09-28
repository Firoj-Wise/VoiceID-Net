# High-Speed Standalone Verification Client

import os
import time
import torch
import numpy as np
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi

class VoiceVerifier:
    """
    High-Speed Standalone Verifier:
    Extracts embeddings and computes cosine similarity for speaker verification / identification.
    """
    def __init__(self, onnx_model_path="voice_engine/voice_encoder.onnx", device="cpu"):
        import onnxruntime as ort
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.sample_rate = 16000

    def extract_fbank(self, wav_path):
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        if sr != self.sample_rate:
            import torchaudio
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
            
        scaled_wave = wav * (1 << 15)
        fbank = kaldi.fbank(
            scaled_wave,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            energy_floor=0.0,
            window_type="hamming",
            sample_frequency=self.sample_rate
        )
        fbank = fbank.unsqueeze(0).transpose(1, 2)
        fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
        return fbank.numpy()

    def get_embedding(self, wav_path):
        """Extracts 192-dim speaker embedding vector."""
        fbank = self.extract_fbank(wav_path)
        ort_inputs = {"fbank": fbank}
        embedding = self.session.run(None, ort_inputs)[0]
        norm_emb = embedding / (np.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-12)
        return norm_emb[0]

    def verify(self, wav1_path, wav2_path, threshold=0.65):
        """
        Computes cosine similarity between two audio utterances.
        Returns: (similarity_score, is_same_speaker)
        """
        e1 = self.get_embedding(wav1_path)
        e2 = self.get_embedding(wav2_path)
        similarity = float(np.dot(e1, e2))
        is_same = similarity >= threshold
        return similarity, is_same
