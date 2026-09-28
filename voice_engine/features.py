# Acoustic Feature Extraction & Augmentation Engine

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

class AudioFeatureExtractor(nn.Module):
    """
    Standard 80-dim log-Mel Filterbank Feature Extractor:
    - 80 Mel bins
    - 25 ms frame length
    - 10 ms frame shift
    - Hamming window, 16kHz sampling rate
    - Cepstral Mean Normalization (CMN)
    """
    def __init__(self, sample_rate=16000, num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0, cmn=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.cmn = cmn

    def extract_fbank_single(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        scaled_wave = waveform * (1 << 15)
        fbank = kaldi.fbank(
            scaled_wave,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=0.0,
            window_type="hamming",
            sample_frequency=self.sample_rate
        )
        fbank = fbank.unsqueeze(0).transpose(1, 2) # (1, 80, num_frames)
        
        if self.cmn:
            fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
            
        return fbank

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        fbanks = []
        for i in range(waveforms.shape[0]):
            fbanks.append(self.extract_fbank_single(waveforms[i]))
        return torch.cat(fbanks, dim=0)


class SpecAugment(nn.Module):
    """
    SpecAugment: Time and Frequency Masking for high acoustic robustness
    """
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        B, C, T = x.shape
        augmented = x.clone()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f_len = torch.randint(0, self.freq_mask_param, (B,), device=x.device)
            f_start = torch.randint(0, max(1, C - self.freq_mask_param), (B,), device=x.device)
            for b in range(B):
                augmented[b, f_start[b]:f_start[b] + f_len[b], :] = 0.0
                
        # Time masking
        if T > self.time_mask_param:
            for _ in range(self.num_time_masks):
                t_len = torch.randint(0, self.time_mask_param, (B,), device=x.device)
                t_start = torch.randint(0, max(1, T - self.time_mask_param), (B,), device=x.device)
                for b in range(B):
                    augmented[b, :, t_start[b]:t_start[b] + t_len[b]] = 0.0
                    
        return augmented
