# VoiceID-Net Representation & Teacher-Guided Distillation Engine (100% Self-Contained)

import os
import sys
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
    Production VoiceID-Net Training Engine:
    - Teacher-Guided Multi-Tier Metric Distillation (Cosine + L2 + Relational Topology)
    - Supervised SubCenterArcFace margin classification
    - Automatic Mixed Precision (AMP) on RTX 4090 GPU
    - SpecAugment for noise invariance
    """
    def __init__(
        self,
        teacher_checkpoint=None,
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

        # 2. Student Backbone Model (Proprietary 2.65M architecture)
        print(f"[Trainer] Initializing VoiceID-Net Student (channels={base_channels}, embed_dim={embed_dim})...")
        self.model = create_model(embed_dim=embed_dim, base_channels=base_channels).to(self.device)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Student Backbone Parameters: {params / 1e6:.2f} M")

        # 3. Pretrained Teacher Model (Frozen for Supervision)
        self.teacher = None
        if teacher_checkpoint and os.path.exists(teacher_checkpoint):
            print(f"[Trainer] Loading Pretrained SOTA Teacher: {teacher_checkpoint}...")
            # Import official loader
            mect_port_path = "/home/oem/wiseyak_backup/wiseai-training-pipeline/packages/trainer-mect-sv/scratch/research-antspeaker"
            if mect_port_path not in sys.path:
                sys.path.append(mect_port_path)
            from mect_port import load_official
            self.teacher, _ = load_official(teacher_checkpoint)
            self.teacher = self.teacher.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print("[Trainer] Pretrained SOTA Teacher loaded and frozen successfully.")

        # 4. Losses
        self.distill_loss_fn = MultiTierMetricLoss().to(self.device)
        self.classifier = None
        if num_classes is not None and self.teacher is None:
            print(f"[Trainer] Attaching SubCenterArcFace classifier for {num_classes} speaker classes.")
            self.classifier = SubCenterArcFace(embed_dim, num_classes, sub_centers=2, s=32.0, m=0.25).to(self.device)

        # 5. Optimizer & Scaler
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

    def train_epoch(self, dataloader, epoch, total_epochs, writer=None, global_step=0):
        self.model.train()
        if self.classifier is not None:
            self.classifier.train()

        total_loss = 0.0
        total_cos_sim = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            wavs = batch["wavs"].to(self.device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(self.device)

            # 1. Extract 80-dim log-Mel filterbanks
            with torch.no_grad():
                fbanks = self.feature_extractor(wavs)

                # Teacher embeddings (Ground-truth supervision)
                if self.teacher is not None:
                    target_embs = self.teacher(fbanks)
                    target_embs = F.normalize(target_embs, p=2, dim=-1)
                else:
                    target_embs = batch.get("target_embs", None)
                    if target_embs is not None:
                        target_embs = target_embs.to(self.device)

            # 2. SpecAugment for student input
            aug_fbanks = self.spec_aug(fbanks)

            # 3. Student Forward pass under AMP
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                model_embs = self.model(aug_fbanks)
                model_embs_unit = F.normalize(model_embs, p=2, dim=-1)

                loss = torch.tensor(0.0, device=self.device)
                cos_sim_val = 0.0

                # Teacher Distillation Loss
                if target_embs is not None:
                    dist_loss, loss_dict = self.distill_loss_fn(model_embs, target_embs)
                    loss = loss + dist_loss
                    cos_sim_val = F.cosine_similarity(model_embs_unit, target_embs, dim=-1).mean().item()

                # Classification Loss (if active)
                if self.classifier is not None and labels is not None:
                    logits = self.classifier(model_embs, labels)
                    loss = loss + F.cross_entropy(logits, labels)

            # 4. Backward & Step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_cos_sim += cos_sim_val
            global_step += 1

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                if target_embs is not None:
                    writer.add_scalar("Distill/Teacher_Cosine_Similarity", cos_sim_val, global_step)

            postfix = {"loss": f"{loss.item():.4f}"}
            if target_embs is not None:
                postfix["sim_to_teacher"] = f"{cos_sim_val:.3f}"
            pbar.set_postfix(postfix)

        n = max(1, len(dataloader))
        return {
            "total_loss": total_loss / n,
            "mean_cosine_similarity": total_cos_sim / n,
            "global_step": global_step
        }

    def save_checkpoint(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "embed_dim": self.model.embed_dim,
            "architecture": "VoiceIDNet"
        }, save_path)
        print(f"[Trainer] Checkpoint saved to: {save_path}")
