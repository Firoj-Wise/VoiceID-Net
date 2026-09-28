# Visualization & Metric Graph Plotting Engine

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path="checkpoints/training_loss_curves.png"):
    """
    Plots high-resolution training loss curves:
    - Total Loss
    - Cosine Similarity Loss
    - L2 Normalized Distance Loss
    - Relational Manifold Loss
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["total"]) + 1)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    plt.plot(epochs, history["total"], "o-", color="#1f77b4", linewidth=2.5, label="Total Loss")
    plt.plot(epochs, history["cos"], "s--", color="#ff7f0e", linewidth=2.0, label="Cosine Alignment Loss")
    plt.plot(epochs, history["l2"], "^-.", color="#2ca02c", linewidth=2.0, label="Normalized L2 Loss")
    plt.plot(epochs, history["rel"], "d:", color="#d62728", linewidth=2.0, label="Relational Manifold Loss")
    
    plt.title("VoiceID-Net Training & Representation Convergence", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12, fontweight="medium")
    plt.ylabel("Loss Magnitude", fontsize=12, fontweight="medium")
    plt.xticks(epochs)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plotting] Training loss curves saved to: {save_path}")


def plot_speaker_separation(same_scores, diff_scores, save_path="checkpoints/speaker_separation_curve.png"):
    """
    Plots speaker score separation distribution and decision threshold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    bins = np.linspace(-1.0, 1.0, 40)
    plt.hist(diff_scores, bins=bins, alpha=0.6, color="#e74c3c", label="Different Speakers (Impostors)", density=True)
    plt.hist(same_scores, bins=bins, alpha=0.6, color="#2ecc71", label="Same Speaker (Genuine)", density=True)
    
    plt.axvline(x=0.60, color="#2c3e50", linestyle="--", linewidth=2, label="Verification Threshold (0.60)")
    
    plt.title("VoiceID-Net Speaker Verification Score Separation", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Cosine Similarity Score", fontsize=12, fontweight="medium")
    plt.ylabel("Probability Density", fontsize=12, fontweight="medium")
    plt.xlim(-1.0, 1.0)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plotting] Speaker separation distribution saved to: {save_path}")
