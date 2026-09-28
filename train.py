#!/usr/bin/env python3
# Training Launch Script for VoiceID-Net with Real-Time TensorBoard Dashboard

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from voice_engine.trainer import VoiceIDTrainer
from voice_engine.dataset import VoxCelebDataset, SpeechDataset, collate_speech_batches
from voice_engine.evaluator import BenchmarkEvaluator
from voice_engine.export_onnx import export_to_onnx, benchmark_latency

def parse_args():
    parser = argparse.ArgumentParser(description="Train VoiceID-Net with Live Dashboard")
    # Dataset paths
    parser.add_argument("--train-dir", type=str,
                        default="/mnt/storage2/wiseai-training-pipeline/raw_data/voxceleb_data/vox2_dev/dev/aac",
                        help="Path to VoxCeleb2 training utterances")
    parser.add_argument("--musan-dir", type=str,
                        default="/mnt/storage2/wiseai-training-pipeline/raw_data/musan",
                        help="Path to MUSAN noise corpus")
    parser.add_argument("--rir-dir", type=str,
                        default="/mnt/storage2/wiseai-training-pipeline/raw_data/RIRS_NOISES",
                        help="Path to RIR reverberation corpus")
    parser.add_argument("--test-wav-dir", type=str,
                        default="/mnt/storage2/wiseai-training-pipeline/raw_data/voxceleb_data/vox1_test",
                        help="Path to VoxCeleb1 benchmark evaluation audio")
    parser.add_argument("--test-trials", type=str,
                        default="/mnt/storage2/wiseai-training-pipeline/raw_data/voxceleb_data/meta/veri_test.txt",
                        help="Path to official VoxCeleb1-O veri_test.txt trial pairs")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Optional pre-computed teacher embedding cache")
    parser.add_argument("--teacher-checkpoint", type=str,
                        default="/home/oem/wiseyak_backup/wiseai-training-pipeline/packages/trainer-mect-sv/scratch/research-antspeaker/checkpoints/mect_b2_vb2.pt",
                        help="Path to pretrained SOTA teacher checkpoint for metric distillation")
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (128 for high VRAM saturation)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (AdamW)")
    parser.add_argument("--chunk-seconds", type=float, default=2.0, help="Chunk length in seconds")
    parser.add_argument("--speed-perturb", action="store_true", default=True, help="3-way speed perturbation")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/voiceid_experiment", help="TensorBoard log directory")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export to ONNX after training")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 75)
    print("VoiceID-Net Production Training Engine (RTX 4090 Accelerated)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Running on: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 0. Live Real-Time Dashboard Logger
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"[Dashboard] Real-Time TensorBoard live logger active at: {args.log_dir}")

    # 1. Dataset Selection
    if args.train_dir and os.path.exists(args.train_dir):
        print(f"[Dataset] Initializing full-scale VoxCeleb2 dataset from: {args.train_dir}")
        dataset = VoxCelebDataset(
            data_dir=args.train_dir,
            musan_dir=args.musan_dir,
            rir_dir=args.rir_dir,
            chunk_seconds=args.chunk_seconds,
            is_train=True,
            speed_perturb=args.speed_perturb
        )
        num_classes = dataset.num_classes
    elif args.cache_file and os.path.exists(args.cache_file):
        print(f"[Dataset] Loading cached representations from: {args.cache_file}")
        cache = torch.load(args.cache_file, map_location="cpu")
        audio_files = [p for p in cache.keys() if os.path.exists(p)]
        target_embeddings = [cache[p] for p in audio_files]
        dataset = SpeechDataset(
            audio_files=audio_files,
            cached_embeddings=target_embeddings,
            chunk_seconds=args.chunk_seconds,
            is_train=True
        )
        num_classes = None
    else:
        raise FileNotFoundError(f"Neither --train-dir nor --cache-file was found.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_speech_batches,
        persistent_workers=(args.num_workers > 0)
    )

    # 2. Benchmark Evaluator (Vox1-O veri_test.txt)
    evaluator = None
    if os.path.exists(args.test_trials) and os.path.exists(args.test_wav_dir):
        print(f"[Evaluator] Initializing Vox1-O benchmark evaluator ({args.test_trials})...")
        evaluator = BenchmarkEvaluator(
            test_wav_dir=args.test_wav_dir,
            trials_path=args.test_trials,
            device=device
        )
        print(f"[Evaluator] Indexed {len(evaluator.trials)} verification trial pairs.")

    # 3. Trainer (Student Backbone + SOTA Teacher Distillation)
    trainer = VoiceIDTrainer(
        teacher_checkpoint=args.teacher_checkpoint,
        num_classes=num_classes,
        embed_dim=192,
        base_channels=64,
        lr=args.lr,
        device=device,
        use_amp=True
    )

    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_eer = float("inf")
    best_loss = float("inf")
    global_step = 0

    print("=" * 75)
    print(f"Training Launched! Batch Size: {args.batch_size} (Optimized VRAM Saturation)")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            dataloader=loader,
            epoch=epoch,
            total_epochs=args.epochs,
            writer=writer,
            global_step=global_step
        )
        global_step = metrics["global_step"]
        epoch_loss = metrics["total_loss"]
        mean_sim = metrics.get("mean_cosine_similarity", 0.0)

        writer.add_scalar("Loss/epoch", epoch_loss, epoch)
        writer.add_scalar("Distill/epoch_mean_sim", mean_sim, epoch)

        # Official Benchmark Evaluation on Vox1-O
        eer_str = "N/A"
        if evaluator is not None and len(evaluator.trials) > 0:
            eval_res = evaluator.evaluate(trainer.model, trainer.feature_extractor)
            curr_eer = eval_res["eer"]
            curr_dcf = eval_res["min_dcf"]
            writer.add_scalar("Benchmark/Vox1_O_EER", curr_eer, epoch)
            writer.add_scalar("Benchmark/Vox1_O_minDCF", curr_dcf, epoch)
            eer_str = f"{curr_eer:.2f}% (minDCF: {curr_dcf:.4f})"

            if curr_eer < best_eer:
                best_eer = curr_eer
                best_path = os.path.join(args.save_dir, "voiceid_best_eer.pt")
                trainer.save_checkpoint(best_path)
                print(f"[Benchmark] New Best Vox1-O EER: {curr_eer:.2f}%! Saved to: {best_path}")

        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Loss: {epoch_loss:.4f} | Sim to Teacher: {mean_sim:.3f} | Vox1-O EER: {eer_str}")

        latest_path = os.path.join(args.save_dir, "voiceid_latest.pt")
        trainer.save_checkpoint(latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trainer.save_checkpoint(os.path.join(args.save_dir, "voiceid_best_loss.pt"))

    writer.close()
    print("=" * 75)
    print(f"[Training Complete] Best Loss: {best_loss:.4f} | Best Vox1-O EER: {best_eer:.2f}%")
    print(f"[Storage] Checkpoints stored in: {args.save_dir}/")
    print("=" * 75)

    if args.export_onnx:
        onnx_path = os.path.join(args.save_dir, "voiceid.onnx")
        print(f"[ONNX] Exporting best model to: {onnx_path}")
        export_to_onnx(trainer.model, output_path=onnx_path)
        benchmark_latency(onnx_path, num_iterations=100)

if __name__ == "__main__":
    main()
