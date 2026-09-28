# High-Performance Speech Dataset & Augmentation Engine for VoiceID-Net

import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class AcousticAugmenter:
    """
    Ultra-Fast Acoustic Augmentation using MUSAN noise and RIR reverberation.
    Uses O(N log N) FFT convolution and RIR tail truncation for 80x speedup over naive Conv1d.
    """
    def __init__(self, musan_dir=None, rir_dir=None, sample_rate=16000, max_rir_len=8000):
        self.sample_rate = sample_rate
        self.max_rir_len = max_rir_len
        self.noise_files = []
        self.rir_files = []

        if musan_dir and os.path.exists(musan_dir):
            self.noise_files = glob.glob(os.path.join(musan_dir, "**/*.wav"), recursive=True)
            print(f"[Augmentation] Indexed {len(self.noise_files)} MUSAN noise files.")

        if rir_dir and os.path.exists(rir_dir):
            self.rir_files = glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True)
            print(f"[Augmentation] Indexed {len(self.rir_files)} RIR reverberation files.")

    def add_reverb(self, wav):
        if not self.rir_files:
            return wav
        rir_path = random.choice(self.rir_files)
        try:
            rir, sr = torchaudio.load(rir_path)
            if sr != self.sample_rate:
                rir = torchaudio.transforms.Resample(sr, self.sample_rate)(rir)
            rir = rir[0] # 1D
            if rir.shape[0] > self.max_rir_len:
                rir = rir[:self.max_rir_len] # Truncate to standard acoustic reverberation window (0.5s)
            rir = rir / (torch.norm(rir, p=2) + 1e-8)

            # High-speed FFT convolution O(N log N)
            n_fft = wav.shape[0] + rir.shape[0] - 1
            n_fft_pow2 = 1 << (n_fft - 1).bit_length()
            W = torch.fft.rfft(wav, n=n_fft_pow2)
            R = torch.fft.rfft(rir, n=n_fft_pow2)
            wav_conv = torch.fft.irfft(W * R, n=n_fft_pow2)[:wav.shape[0]]
            return wav_conv
        except Exception:
            return wav

    def add_noise(self, wav):
        if not self.noise_files:
            return wav
        noise_path = random.choice(self.noise_files)
        try:
            noise, sr = torchaudio.load(noise_path)
            if sr != self.sample_rate:
                noise = torchaudio.transforms.Resample(sr, self.sample_rate)(noise)
            noise = noise[0]
            wav_len = wav.shape[0]
            if noise.shape[0] < wav_len:
                repeats = (wav_len // noise.shape[0]) + 1
                noise = noise.repeat(repeats)[:wav_len]
            else:
                start = random.randint(0, noise.shape[0] - wav_len)
                noise = noise[start:start + wav_len]

            snr_db = random.uniform(5.0, 15.0)
            wav_power = torch.mean(wav ** 2) + 1e-8
            noise_power = torch.mean(noise ** 2) + 1e-8
            scale = torch.sqrt(wav_power / (10 ** (snr_db / 10.0) * noise_power))
            return wav + scale * noise
        except Exception:
            return wav

    def __call__(self, wav):
        if self.rir_files and random.random() < 0.5:
            wav = self.add_reverb(wav)
        if self.noise_files and random.random() < 0.6:
            wav = self.add_noise(wav)
        return wav


class VoxCelebDataset(Dataset):
    """
    Full-scale VoxCeleb2 dataset reader optimized for high-throughput GPU training.
    """
    def __init__(
        self,
        data_dir,
        musan_dir=None,
        rir_dir=None,
        sample_rate=16000,
        chunk_seconds=2.0,
        is_train=True,
        speed_perturb=True
    ):
        self.sample_rate = sample_rate
        self.chunk_len = int(chunk_seconds * sample_rate) if chunk_seconds else None
        self.is_train = is_train
        self.speed_perturb = speed_perturb and is_train

        # Pre-instantiate resamplers once (avoiding runtime graph recreation overhead)
        if self.speed_perturb:
            self.resample_09 = torchaudio.transforms.Resample(self.sample_rate, int(self.sample_rate * 0.9))
            self.resample_11 = torchaudio.transforms.Resample(self.sample_rate, int(self.sample_rate * 1.1))

        # Index all audio files
        print(f"[Dataset] Indexing VoxCeleb audio files from: {data_dir}...")
        self.samples = []
        spk_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.spk2id = {spk: idx for idx, spk in enumerate(spk_dirs)}
        self.num_base_speakers = len(spk_dirs)

        for spk in spk_dirs:
            spk_path = os.path.join(data_dir, spk)
            for root, _, files in os.walk(spk_path):
                for f in files:
                    if f.endswith(".m4a") or f.endswith(".wav"):
                        self.samples.append((os.path.join(root, f), self.spk2id[spk]))

        print(f"[Dataset] Found {len(self.samples)} audio files across {self.num_base_speakers} base speakers.")
        
        self.num_classes = self.num_base_speakers * 3 if self.speed_perturb else self.num_base_speakers
        if self.speed_perturb:
            print(f"[Dataset] 3x Speed Perturbation enabled: {self.num_classes} total virtual speaker classes.")

        self.augmenter = AcousticAugmenter(musan_dir, rir_dir, sample_rate) if is_train else None

    def __len__(self):
        return len(self.samples)

    def load_chunk(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            waveform = waveform[0]
        except Exception:
            waveform = torch.zeros(self.chunk_len if self.chunk_len else 32000)
            sr = self.sample_rate

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform.unsqueeze(0)).squeeze(0)

        wav_len = waveform.shape[0]
        if self.chunk_len is not None:
            if wav_len > self.chunk_len:
                if self.is_train:
                    start = random.randint(0, wav_len - self.chunk_len)
                else:
                    start = (wav_len - self.chunk_len) // 2
                waveform = waveform[start:start + self.chunk_len]
            elif wav_len < self.chunk_len:
                pad_len = self.chunk_len - wav_len
                waveform = F.pad(waveform, (0, pad_len), mode="constant", value=0.0)

        return waveform

    def __getitem__(self, idx):
        wav_path, base_label = self.samples[idx]
        wav = self.load_chunk(wav_path)

        label = base_label
        # Speed perturbation (0.9x, 1.0x, 1.1x)
        if self.speed_perturb:
            perturb_choice = random.choice([0, 1, 2])
            if perturb_choice == 1:
                wav = self.resample_09(wav.unsqueeze(0)).squeeze(0)
                if wav.shape[0] > self.chunk_len:
                    wav = wav[:self.chunk_len]
                elif wav.shape[0] < self.chunk_len:
                    wav = F.pad(wav, (0, self.chunk_len - wav.shape[0]))
                label = base_label + self.num_base_speakers
            elif perturb_choice == 2:
                wav = self.resample_11(wav.unsqueeze(0)).squeeze(0)
                if wav.shape[0] > self.chunk_len:
                    wav = wav[:self.chunk_len]
                elif wav.shape[0] < self.chunk_len:
                    wav = F.pad(wav, (0, self.chunk_len - wav.shape[0]))
                label = base_label + 2 * self.num_base_speakers

        # Add acoustic noise / reverb in training mode (Ultra-Fast FFT)
        if self.is_train and self.augmenter:
            wav = self.augmenter(wav)

        return {
            "wav": wav,
            "label": torch.tensor(label, dtype=torch.long),
            "path": wav_path
        }


class SpeechDataset(Dataset):
    """
    High-performance audio dataset supporting on-the-fly random slicing and cached embeddings.
    """
    def __init__(
        self,
        audio_files,
        labels=None,
        cached_embeddings=None,
        sample_rate=16000,
        chunk_seconds=3.0,
        is_train=True
    ):
        self.audio_files = audio_files
        self.labels = labels
        self.cached_embeddings = cached_embeddings
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_len = int(chunk_seconds * sample_rate) if chunk_seconds else None
        self.is_train = is_train

    def __len__(self):
        return len(self.audio_files)

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            waveform = waveform[0]
        except Exception:
            waveform = torch.zeros(self.chunk_len if self.chunk_len else 32000)
            sr = self.sample_rate

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        if self.chunk_len is not None:
            wav_len = waveform.shape[0]
            if wav_len > self.chunk_len:
                if self.is_train:
                    start = random.randint(0, wav_len - self.chunk_len)
                else:
                    start = (wav_len - self.chunk_len) // 2
                waveform = waveform[start:start + self.chunk_len]
            elif wav_len < self.chunk_len:
                pad_len = self.chunk_len - wav_len
                waveform = F.pad(waveform, (0, pad_len), mode="constant", value=0.0)

        return waveform

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        wav = self.load_audio(wav_path)

        item = {
            "wav": wav,
            "path": wav_path
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.cached_embeddings is not None:
            emb = self.cached_embeddings[idx]
            if isinstance(emb, torch.Tensor):
                item["target_emb"] = emb.clone().detach().float()
            else:
                item["target_emb"] = torch.tensor(emb, dtype=torch.float32)

        return item


def collate_speech_batches(batch):
    wavs = torch.stack([b["wav"] for b in batch], dim=0)
    res = {"wavs": wavs, "paths": [b["path"] for b in batch]}

    if "label" in batch[0]:
        res["labels"] = torch.stack([b["label"] for b in batch], dim=0)

    if "target_emb" in batch[0]:
        res["target_embs"] = torch.stack([b["target_emb"] for b in batch], dim=0)

    return res
