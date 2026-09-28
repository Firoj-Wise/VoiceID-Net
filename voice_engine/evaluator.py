# Official VoxCeleb Benchmark Evaluator (EER% and minDCF)

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torchaudio

def compute_eer(scores, labels):
    """
    Computes Equal Error Rate (EER) and the optimal verification threshold.
    scores: numpy array of cosine similarities [-1, 1]
    labels: numpy array of binary ground-truth {0, 1}
    """
    # Sort scores in descending order
    desc_order = np.argsort(scores)[::-1]
    scores_sorted = scores[desc_order]
    labels_sorted = labels[desc_order]

    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)

    if total_pos == 0 or total_neg == 0:
        return 0.0, 0.0

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)

    fn = total_pos - tp
    tn = total_neg - fp

    fpr = fp / total_neg
    fnr = fn / total_pos

    # Find the point where FPR and FNR are closest
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)

    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    threshold = scores_sorted[idx]
    return eer, threshold

def compute_min_dcf(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """
    Computes minimum Decision Cost Function (minDCF).
    Standard VoxCeleb evaluation parameters: p_target=0.01, c_miss=1, c_fa=1
    """
    desc_order = np.argsort(scores)[::-1]
    scores_sorted = scores[desc_order]
    labels_sorted = labels[desc_order]

    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)

    if total_pos == 0 or total_neg == 0:
        return 0.0

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)
    fn = total_pos - tp

    fpr = fp / total_neg
    fnr = fn / total_pos

    c_det = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = np.min(c_det) / c_def
    return float(min_dcf)

class BenchmarkEvaluator:
    """
    Evaluator for official VoxCeleb trial lists (veri_test.txt, list_test_hard.txt, etc.)
    """
    def __init__(self, test_wav_dir, trials_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.test_wav_dir = test_wav_dir
        self.trials_path = trials_path
        self.device = torch.device(device)
        self.trials = self.load_trials(trials_path)

    def load_trials(self, path):
        trials = []
        if not os.path.exists(path):
            print(f"[Warning] Trial file '{path}' not found.")
            return trials
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # format: label wav1 wav2 (e.g. 1 id10270/x.wav id10270/y.wav)
                    label = int(parts[0])
                    w1 = parts[1]
                    w2 = parts[2]
                    trials.append((label, w1, w2))
        return trials

    def extract_embedding(self, model, feature_extractor, wav_path):
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        waveform = torch.from_numpy(waveform).to(self.device)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        with torch.no_grad():
            fbank = feature_extractor(waveform.unsqueeze(0))
            emb = model(fbank)
            emb = F.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0).cpu().numpy()

    def evaluate(self, model, feature_extractor):
        model.eval()
        if not self.trials:
            return {"eer": 0.0, "min_dcf": 0.0, "threshold": 0.0}

        # Cache unique test files
        unique_files = set()
        for _, w1, w2 in self.trials:
            unique_files.add(w1)
            unique_files.add(w2)

        emb_cache = {}
        for rel_path in tqdm(unique_files, desc="[Eval] Extracting Test Embeddings", leave=False):
            # Check different path options (direct or prefixed)
            full_path = os.path.join(self.test_wav_dir, rel_path)
            if not os.path.exists(full_path):
                # Search if nested inside wav/
                nested = os.path.join(self.test_wav_dir, "wav", rel_path)
                if os.path.exists(nested):
                    full_path = nested

            if os.path.exists(full_path):
                emb_cache[rel_path] = self.extract_embedding(model, feature_extractor, full_path)

        scores = []
        labels = []
        for label, w1, w2 in self.trials:
            if w1 in emb_cache and w2 in emb_cache:
                e1 = emb_cache[w1]
                e2 = emb_cache[w2]
                cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
                scores.append(cos_sim)
                labels.append(label)

        if not scores:
            return {"eer": 0.0, "min_dcf": 0.0, "threshold": 0.0}

        scores = np.array(scores)
        labels = np.array(labels)

        eer, thresh = compute_eer(scores, labels)
        min_dcf = compute_min_dcf(scores, labels)

        return {
            "eer": eer,
            "min_dcf": min_dcf,
            "threshold": thresh,
            "num_trials": len(scores)
        }
