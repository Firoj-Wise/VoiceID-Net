# VoiceID-Net Representation & Supervised Training Engine (100% Self-Contained)

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from voice_engine.models import create_model
from voice_engine.features import AudioFeatureExtractor, SpecAugment
from voice_engine.losses import MultiTierMetricLoss, SubCenterArcFace

class VoiceIDTrainer:
    """
    Self-Contained VoiceID-Net Training Engine:
    - Supervised SubCenterArcFace classification across VoxCeleb speakers
    - Multi-Tier Metric Alignment distillation when target embeddings are present
    - Automatic Mixed Precision (AMP) on RTX 4090 GPU
    - SpecAugment for acoustic robustness
    """
    def __init__(
        self,
        num_classes=None,
        embed_dim=192,
        base_channels=64,
        lr=1e-3,
        weight_decay=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True
    ):
        self.device = torch.device(device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.num_classes = num_classes

        # 1. Feature extractor & SpecAugment
        self.feature_extractor = AudioFeatureExtractor().to(self.device)
        self.spec_aug = SpecAugment().to(self.device)

        # 2. Backbone Model
        print(f"[Trainer] Initializing VoiceID-Net (channels={base_channels}, embed_dim={embed_dim})...")
        self.model = create_model(embed_dim=embed_dim, base_channels=base_channels).to(self.device)

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model Backbone Parameters: {params / 1e6:.2f} M")

        # 3. Supervised Classification Head & Metric Distillation
        self.classifier = None
        if num_classes is not None:
            print(f"[Trainer] Attaching SubCenterArcFace classifier for {num_classes} speaker classes.")
            self.classifier = SubCenterArcFace(embed_dim, num_classes, sub_centers=2, s=32.0, m=0.25).to(self.device)

        self.distill_loss_fn = MultiTierMetricLoss().to(self.device)

        # 4. Optimizer & Scaler
        trainable_params = list(self.model.parameters())
        if self.classifier is not None:
            trainable_params += list(self.classifier.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        if self.classifier is not None:
            self.classifier.train()

        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}]", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            wavs = batch["wavs"].to(self.device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(self.device)
            target_embs = batch.get("target_embs", None)
            if target_embs is not None:
                target_embs = target_embs.to(self.device)

            # 1. Extract 80-dim log-Mel filterbanks
            with torch.no_grad():
                fbanks = self.feature_extractor(wavs)

            # 2. SpecAugment
            aug_fbanks = self.spec_aug(fbanks)

            # 3. Forward pass under AMP
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                model_embs = self.model(aug_fbanks)

                loss = torch.tensor(0.0, device=self.device)
                acc = 0.0

                # Classification Loss (ArcFace)
                if self.classifier is not None and labels is not None:
                    logits = self.classifier(model_embs, labels)
                    cls_loss = F.cross_entropy(logits, labels)
                    loss = loss + cls_loss
                    acc = (logits.argmax(dim=-1) == labels).float().mean().item()

                # Distillation Loss (Cosine + L2 + Relational Topology)
                if target_embs is not None:
                    dist_loss, _ = self.distill_loss_fn(model_embs, target_embs)
                    loss = loss + dist_loss

            # 4. Backward & Step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_acc += acc

            postfix = {"loss": f"{loss.item():.4f}"}
            if self.classifier is not None:
                postfix["acc"] = f"{acc * 100:.1f}%"
            pbar.set_postfix(postfix)

        n = max(1, len(dataloader))
        return {
            "total_loss": total_loss / n,
            "accuracy": total_acc / n if self.classifier is not None else 0.0
        }

    def save_checkpoint(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "embed_dim": self.model.embed_dim,
            "architecture": "VoiceIDNet"
        }, save_path)
        print(f"[Trainer] Checkpoint saved to: {save_path}")
