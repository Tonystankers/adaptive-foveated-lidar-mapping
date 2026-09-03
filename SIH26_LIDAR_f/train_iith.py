import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from segmentation_engine import PointNetPPClassifier


IITH_ROOT = r"C:\Users\Acer\Downloads\IITH_LiDAR_ground_dataset_labelled_raw (2)\IITH_LiDAR_ground_dataset_labelled_raw"
LABEL_DIR = os.path.join(IITH_ROOT, "labelled_data")

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "pointnet_pp_iith_2class.pt"
)


def decode_rgb_uint32(rgb_value):
    rgb = int(rgb_value)
    r = (rgb >> 16) & 255
    g = (rgb >> 8) & 255
    b = rgb & 255
    return r, g, b


def read_iith_pcd(path):
    points = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        data_started = False

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("DATA"):
                data_started = True
                continue

            if not data_started:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                rgb = float(parts[3])

                r, g, b = decode_rgb_uint32(rgb)

                points.append([x, y, z])

                # IITH:
                # RED   = Ground
                # GREEN = Non-ground
                if r > g and r > b:
                    labels.append(0)       # Ground
                else:
                    labels.append(1)       # Non-ground

            except ValueError:
                continue

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


class IITHDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        points, labels = read_iith_pcd(self.files[idx])

        # Normalize each frame.
        # Center X/Y and normalize scale.
        center = points.mean(axis=0)
        points = points - center

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        # XYZ + intensity
        intensity = np.full(
            (len(points), 1),
            0.5,
            dtype=np.float32
        )

        xyzi = np.concatenate(
            [points, intensity],
            axis=1
        ).astype(np.float32)

        return (
            torch.from_numpy(xyzi),
            torch.from_numpy(labels)
        )


def main():

    print("=" * 70)
    print("IITH LiDAR - 2 CLASS POINTNET++ TRAINING")
    print("=" * 70)

    files = sorted(
        glob.glob(os.path.join(LABEL_DIR, "*.pcd"))
    )

    print("Files found:", len(files))

    if len(files) < 3:
        print("ERROR: Not enough IITH files.")
        return

    # First 9 files for training
    # Last 2 files for validation
    train_files = files[:-2]
    val_files = files[-2:]

    print()
    print("Training files:", len(train_files))
    for f in train_files:
        print("  ", os.path.basename(f))

    print()
    print("Validation files:", len(val_files))
    for f in val_files:
        print("  ", os.path.basename(f))

    train_ds = IITHDataset(train_files)
    val_ds = IITHDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print()
    print("[*] Creating 2-class PointNet++ model...")

    model = PointNetPPClassifier(
        in_channels=4,
        num_classes=2
    )

    device = "cpu"
    model = model.to(device)

    # Stronger weight for the minority non-ground class.
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, 4.0],
            dtype=torch.float32
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    epochs = 30

    best_f1 = -1

    print()
    print("Starting training...")
    print()

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0.0

        for xyzi, labels in train_loader:

            xyzi = xyzi.transpose(1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(xyzi)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0
            )

            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()

        total = 0
        correct = 0

        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():

            for xyzi, labels in val_loader:

                xyzi = xyzi.transpose(1, 2).to(device)
                labels = labels.to(device)

                logits = model(xyzi)

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                total += labels.numel()

                correct += (
                    predictions == labels
                ).sum().item()

                tp += (
                    (predictions == 1) &
                    (labels == 1)
                ).sum().item()

                fp += (
                    (predictions == 1) &
                    (labels == 0)
                ).sum().item()

                fn += (
                    (predictions == 0) &
                    (labels == 1)
                ).sum().item()

        accuracy = correct / max(total, 1)

        precision = tp / max(tp + fp, 1)

        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall /
            max(precision + recall, 1e-8)
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={train_loss / len(train_loader):.4f} | "
            f"accuracy={accuracy * 100:.2f}% | "
            f"precision={precision * 100:.2f}% | "
            f"recall={recall * 100:.2f}% | "
            f"F1={f1 * 100:.2f}%"
        )

        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                CHECKPOINT
            )

            print("  -> BEST MODEL SAVED")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("Best validation F1:", f"{best_f1 * 100:.2f}%")
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
