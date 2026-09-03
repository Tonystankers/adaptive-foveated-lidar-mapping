"""
train_model.py - Supervised training loop for the PointNet++-style semantic
segmentation network used by segmentation_engine.PointNetPPSegmentationEngine.
SIH26053: Adaptive Variable Resolution 2.5D LiDAR Mapping Pipeline

Why this exists:
    The brief calls for a *trained* deep-learning model that performs semantic
    segmentation of point clouds into terrain / static-obstacle sub-classes /
    dynamic-object sub-classes. Ground-truth labels for real LiDAR data
    (e.g. SemanticKITTI) require downloading a multi-GB external dataset which
    is out of scope here, so this script trains on labeled point clouds
    produced by data_loader.SyntheticLidarGenerator instead - it generates
    physically-plausible scenes (roads, buildings, poles, trees, vehicles,
    pedestrians) WITH ground-truth per-point class labels, so the model is
    trained with genuine supervision rather than shipped as an untrained
    skeleton. The same model + training code works unchanged against real
    labeled LiDAR (e.g. SemanticKITTI's .label files) if you point
    SemanticKittiDataset at a local copy - see --kitti-dir.

Usage:
    python train_model.py --epochs 20 --points-per-frame 4096
    python train_model.py --kitti-dir ./dataset --epochs 20   # train on real data instead

Output:
    checkpoints/pointnet_pp_7class.pt   (model weights, loaded automatically
                                          by PointNetPPSegmentationEngine)
    checkpoints/training_report.json    (loss curve, final val metrics)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np

from data_loader import SemanticClass, SemanticKittiDataset, SyntheticLidarGenerator
from evaluation import compute_confusion_matrix, metrics_from_confusion_matrix, format_metrics_report

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from segmentation_engine import PointNetPPClassifier

NUM_CLASSES = SemanticClass.num_classes()
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "pointnet_pp_7class.pt")
REPORT_PATH = os.path.join(CHECKPOINT_DIR, "training_report.json")


def sample_fixed_points(points: np.ndarray, labels: np.ndarray, intensities: np.ndarray,
                         n_points: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws a fixed-size (n_points,) subset so batches can be stacked into a
    dense tensor. Uses random sampling with replacement only when the frame
    has fewer raw points than n_points (rare - only for very sparse frames).
    """
    n_available = len(points)
    if n_available >= n_points:
        idx = rng.choice(n_available, size=n_points, replace=False)
    else:
        idx = rng.choice(n_available, size=n_points, replace=True)
    xyzi = np.column_stack([points[idx], intensities[idx]]).astype(np.float32)
    return xyzi, labels[idx].astype(np.int64)


if TORCH_AVAILABLE:
    class SyntheticPointCloudDataset(Dataset):
        """
        On-the-fly synthetic LiDAR scene generator wrapped as a PyTorch Dataset.
        A fresh random scene (different object counts/positions/timestamps) is
        generated per index so the model sees varied layouts across epochs.
        """
        def __init__(self, n_scenes: int, points_per_frame: int, seed: int, max_range: float = 100.0):
            self.n_scenes = n_scenes
            self.points_per_frame = points_per_frame
            self.max_range = max_range
            self.base_seed = seed

        def __len__(self) -> int:
            return self.n_scenes

        def __getitem__(self, idx: int):
            # Distinct seed per sample -> distinct random scene layout/timestamp
            gen = SyntheticLidarGenerator(seed=self.base_seed + idx, max_range=self.max_range)
            timestamp = float((idx % 50)) * 0.1
            n_cars = 4 + (idx % 9)
            n_peds = 2 + (idx % 7)
            n_trees = 6 + (idx % 12)
            n_bldgs = 3 + (idx % 6)
            frame = gen.generate_frame(
                frame_id=idx, timestamp=timestamp,
                num_cars=n_cars, num_pedestrians=n_peds,
                num_trees=n_trees, num_buildings=n_bldgs,
            )
            rng = np.random.default_rng(self.base_seed + idx)
            xyzi, labels = sample_fixed_points(
                frame.points, frame.labels, frame.intensities, self.points_per_frame, rng
            )
            return torch.from_numpy(xyzi), torch.from_numpy(labels)


class KittiPointCloudDataset:
    """Wraps SemanticKittiDataset with fixed-size point sampling, for training on real labeled LiDAR."""
    def __init__(self, root_dir: str, sequence_id: str, points_per_frame: int, seed: int = 0):
        self.ds = SemanticKittiDataset(root_dir=root_dir, sequence_id=sequence_id, load_labels=True)
        self.points_per_frame = points_per_frame
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        frame = self.ds[idx]
        xyzi, labels = sample_fixed_points(frame.points, frame.labels, frame.intensities,
                                            self.points_per_frame, self.rng)
        return torch.from_numpy(xyzi), torch.from_numpy(labels)


def compute_class_weights(label_batches: List[np.ndarray], num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights to counter the heavy ground/vegetation imbalance."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for lbl in label_batches:
        counts += np.bincount(lbl.reshape(-1), minlength=num_classes)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def train(
    epochs: int = 15,
    points_per_frame: int = 4096,
    batch_size: int = 4,
    train_scenes: int = 160,
    val_scenes: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    kitti_dir: str = None,
    kitti_sequence: str = "00",
    device: str = "cpu",
) -> dict:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train the model (pip install torch).")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 80)
    print("  SIH26053: Training PointNet++-style Semantic Segmentation Model (7-class)")
    print("=" * 80)

    if kitti_dir and os.path.isdir(kitti_dir):
        print(f"[*] Data source: real labeled LiDAR at '{kitti_dir}' (sequence {kitti_sequence})")
        full_ds = KittiPointCloudDataset(kitti_dir, kitti_sequence, points_per_frame, seed=seed)
        n_total = len(full_ds)
        n_val = max(1, int(0.15 * n_total))
        val_idx = list(range(n_total))[-n_val:]
        train_idx = list(range(n_total))[:-n_val]

        class _Subset(Dataset):
            def __init__(self, base, idxs):
                self.base, self.idxs = base, idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.base[self.idxs[i]]

        train_ds = _Subset(full_ds, train_idx)
        val_ds = _Subset(full_ds, val_idx)
    else:
        print(f"[*] Data source: synthetic generator ({train_scenes} train / {val_scenes} val scenes)")
        train_ds = SyntheticPointCloudDataset(train_scenes, points_per_frame, seed=seed)
        val_ds = SyntheticPointCloudDataset(val_scenes, points_per_frame, seed=seed + 100000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Estimate class weights from a handful of training batches (cheap, no need to scan everything)
    print("[*] Estimating class weights from a sample of training scenes...")
    sample_labels = []
    for i, (_, lbl) in enumerate(train_loader):
        sample_labels.append(lbl.numpy())
        if i >= 15:
            break
    class_weights = compute_class_weights(sample_labels, NUM_CLASSES)
    print(f"    Class weights: {dict(zip([SemanticClass.get_name(c) for c in range(NUM_CLASSES)], np.round(class_weights, 2)))}")

    model = PointNetPPClassifier(in_channels=4, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    t_start = time.time()

    best_val_acc = -1.0
    best_state = None
    best_cm = None
    best_preds_gts = (None, None)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xyzi, labels in train_loader:
            xyzi = xyzi.transpose(1, 2).to(device)   # (B, N, 4) -> (B, 4, N)
            labels = labels.to(device)                # (B, N)

            optimizer.zero_grad()
            logits = model(xyzi)                      # (B, num_classes, N)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds, all_gts = [], []
        with torch.no_grad():
            for xyzi, labels in val_loader:
                xyzi_t = xyzi.transpose(1, 2).to(device)
                labels_t = labels.to(device)
                logits = model(xyzi_t)
                loss = criterion(logits, labels_t)
                val_loss_sum += loss.item()
                val_batches += 1
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds.reshape(-1))
                all_gts.append(labels.numpy().reshape(-1))

        val_loss = val_loss_sum / max(val_batches, 1)
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)
        val_acc = float(np.mean(all_preds == all_gts))
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_cm = compute_confusion_matrix(all_gts, all_preds, NUM_CLASSES)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | lr={current_lr:.2e}"
              f"{'  <- best' if is_best else ''}")

    elapsed = time.time() - t_start
    print(f"\n[+] Training complete in {elapsed:.1f}s. Best val_acc={best_val_acc*100:.2f}% "
          f"(restoring best-epoch weights for the saved checkpoint).")

    # Restore best-epoch weights before saving/evaluating, since per-epoch
    # validation accuracy is noisy on a small val set and the last epoch is
    # not necessarily the best one.
    model.load_state_dict(best_state)
    metrics = metrics_from_confusion_matrix(best_cm)
    print("\n" + format_metrics_report(metrics, title="FINAL VALIDATION METRICS (Best Epoch, Synthetic Held-Out Scenes)"))

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n[+] Saved trained checkpoint to: {CHECKPOINT_PATH}")

    report = {
        "epochs": epochs,
        "points_per_frame": points_per_frame,
        "batch_size": batch_size,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "training_time_sec": elapsed,
        "history": history,
        "best_val_accuracy": best_val_acc,
        "best_val_confusion_matrix": best_cm.tolist(),
        "best_val_metrics": metrics,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved training report to: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PointNet++-style 7-class segmentation model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--points-per-frame", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-scenes", type=int, default=160)
    parser.add_argument("--val-scenes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kitti-dir", type=str, default=None, help="Path to real SemanticKITTI root (optional)")
    parser.add_argument("--kitti-sequence", type=str, default="00")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        points_per_frame=args.points_per_frame,
        batch_size=args.batch_size,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        lr=args.lr,
        seed=args.seed,
        kitti_dir=args.kitti_dir,
        kitti_sequence=args.kitti_sequence,
        device=args.device,
    )
