"""
Train PointNet++ on REAL nuScenes lidarseg data.

Uses the actual extracted nuScenes lidarseg labels.

Existing checkpoint is NOT overwritten.

New checkpoint:
    checkpoints/pointnet_pp_7class_nuscenes.pt
"""

import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from segmentation_engine import PointNetPPClassifier


# ============================================================
# PATHS
# ============================================================

NUSCENES = r"C:\Users\Acer\Downloads\nuscenes_raw"

LIDARSEG = r"C:\Users\Acer\Downloads\nuScenes-lidarseg-mini-v1.0\extracted"

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints"
)

OUTPUT_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "pointnet_pp_7class_nuscenes.pt"
)

OUTPUT_REPORT = os.path.join(
    CHECKPOINT_DIR,
    "nuscenes_training_report.json"
)


# ============================================================
# YOUR 7 CLASSES
# ============================================================

CLASS_NAMES = [
    "Noise",
    "Ground",
    "Static Structure",
    "Static Pole",
    "Static Vegetation",
    "Dynamic Vehicle",
    "Dynamic Pedestrian",
]


# ============================================================
# nuScenes 32 classes -> YOUR 7 CLASSES
# ============================================================

NUSCENES_TO_7 = {

    # Noise
    0: 0,

    # Pedestrians / humans
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 6,
    6: 6,
    7: 6,
    8: 6,

    # Static objects
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,

    # Vehicles
    14: 5,
    15: 5,
    16: 5,
    17: 5,
    18: 5,
    19: 5,
    20: 5,
    21: 5,
    22: 5,
    23: 5,

    # Ground / surfaces
    24: 1,
    25: 1,
    26: 1,
    27: 1,

    # Structures
    28: 2,
    29: 2,

    # Vegetation
    30: 4,

    # Ego vehicle
    31: 2,
}


# ============================================================
# DATASET
# ============================================================

class NuScenesDataset(Dataset):

    def __init__(
        self,
        num_points=4096,
        seed=42
    ):

        self.num_points = num_points
        self.seed = seed

        sample_data_path = os.path.join(
            NUSCENES,
            "v1.0-mini",
            "sample_data.json"
        )

        lidarseg_json_path = os.path.join(
            LIDARSEG,
            "v1.0-mini",
            "lidarseg.json"
        )

        print("Loading nuScenes metadata...")

        if not os.path.exists(sample_data_path):
            raise FileNotFoundError(
                f"sample_data.json not found:\n{sample_data_path}"
            )

        if not os.path.exists(lidarseg_json_path):
            raise FileNotFoundError(
                f"lidarseg.json not found:\n{lidarseg_json_path}"
            )

        with open(
            sample_data_path,
            "r",
            encoding="utf-8"
        ) as f:
            sample_data = json.load(f)

        with open(
            lidarseg_json_path,
            "r",
            encoding="utf-8"
        ) as f:
            lidarseg = json.load(f)

        label_map = {
            x["sample_data_token"]: x["filename"]
            for x in lidarseg
        }

        # Only real LIDAR_TOP keyframes
        records = [
            x
            for x in sample_data
            if "LIDAR_TOP" in x["filename"]
            and x["is_key_frame"]
            and x["token"] in label_map
        ]

        self.records = records
        self.label_map = label_map

        print(
            f"Found {len(self.records)} "
            f"labeled LIDAR_TOP keyframes."
        )

        if len(self.records) == 0:
            raise RuntimeError(
                "No labeled nuScenes LIDAR_TOP frames found."
            )


    def __len__(self):
        return len(self.records)


    # ========================================================
    # FIND ACTUAL LABEL FILE
    # ========================================================

    def get_label_file(self, idx):

        record = self.records[idx]

        label_relative = self.label_map[
            record["token"]
        ]

        label_relative = label_relative.replace(
            "/",
            os.sep
        )

        # Possible metadata path:
        # lidarseg/v1.0-mini/xxxxx_lidarseg.bin

        # Remove leading "lidarseg\"
        prefix = "lidarseg" + os.sep

        if label_relative.lower().startswith(
            prefix.lower()
        ):
            label_relative = label_relative[
                len(prefix):
            ]

        # Now label_relative should be:
        # v1.0-mini\xxxxx_lidarseg.bin

        # If it still starts with v1.0-mini,
        # remove that because LIDARSEG already points
        # to the extracted folder.
        prefix2 = "v1.0-mini" + os.sep

        if label_relative.lower().startswith(
            prefix2.lower()
        ):
            label_relative = label_relative[
                len(prefix2):
            ]

        # Actual location:
        #
        # nuScenes-lidarseg-mini-v1.0
        #     extracted
        #         lidarseg
        #             v1.0-mini
        #                 file.bin
        #
        # So first try the correct real location.

        candidate1 = os.path.join(
            LIDARSEG,
            "lidarseg",
            "v1.0-mini",
            os.path.basename(label_relative)
        )

        # Second possible extracted layout
        candidate2 = os.path.join(
            LIDARSEG,
            "v1.0-mini",
            os.path.basename(label_relative)
        )

        # Third possible layout
        candidate3 = os.path.join(
            LIDARSEG,
            os.path.basename(label_relative)
        )

        candidates = [
            candidate1,
            candidate2,
            candidate3
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "\nCould not find lidarseg label file.\n"
            f"Frame: {idx}\n"
            f"Original metadata path: "
            f"{self.label_map[record['token']]}\n\n"
            f"Tried:\n"
            + "\n".join(candidates)
        )


    # ========================================================
    # LOAD FRAME
    # ========================================================

    def __getitem__(self, idx):

        record = self.records[idx]

        lidar_file = os.path.join(
            NUSCENES,
            record["filename"].replace(
                "/",
                os.sep
            )
        )

        label_file = self.get_label_file(idx)

        # ----------------------------------------------------
        # Check LiDAR file
        # ----------------------------------------------------

        if not os.path.exists(lidar_file):

            raise FileNotFoundError(
                f"\nLiDAR file not found:\n{lidar_file}"
            )

        # ----------------------------------------------------
        # nuScenes LiDAR:
        #
        # x, y, z, intensity, ring
        # ----------------------------------------------------

        points = np.fromfile(
            lidar_file,
            dtype=np.float32
        ).reshape(-1, 5)

        raw_labels = np.fromfile(
            label_file,
            dtype=np.uint8
        )

        if len(points) != len(raw_labels):

            raise RuntimeError(
                f"\nPoint/label mismatch in frame {idx}: "
                f"{len(points)} points vs "
                f"{len(raw_labels)} labels"
            )

        # ----------------------------------------------------
        # Convert 32 nuScenes classes -> 7 classes
        # ----------------------------------------------------

        labels = np.array(
            [
                NUSCENES_TO_7.get(
                    int(x),
                    0
                )
                for x in raw_labels
            ],
            dtype=np.int64
        )

        # Model uses:
        # x, y, z, intensity

        xyzi = points[:, :4]

        # ----------------------------------------------------
        # Fixed-size point sampling
        # ----------------------------------------------------

        rng = np.random.default_rng(
            self.seed + idx
        )

        if len(xyzi) >= self.num_points:

            indices = rng.choice(
                len(xyzi),
                size=self.num_points,
                replace=False
            )

        else:

            indices = rng.choice(
                len(xyzi),
                size=self.num_points,
                replace=True
            )

        xyzi = xyzi[indices]
        labels = labels[indices]

        return (
            torch.from_numpy(
                xyzi.astype(np.float32)
            ),
            torch.from_numpy(
                labels.astype(np.int64)
            )
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    print("\nCalculating class weights...")

    counts = np.zeros(
        7,
        dtype=np.float64
    )

    # Scan all frames once
    for i in range(len(dataset)):

        if (i + 1) % 25 == 0 or i == 0:

            print(
                f"\rScanning labels: "
                f"{i + 1}/{len(dataset)}",
                end=""
            )

        label_file = dataset.get_label_file(i)

        raw_labels = np.fromfile(
            label_file,
            dtype=np.uint8
        )

        mapped = np.array(
            [
                NUSCENES_TO_7.get(
                    int(x),
                    0
                )
                for x in raw_labels
            ],
            dtype=np.int64
        )

        counts += np.bincount(
            mapped,
            minlength=7
        )

    print()

    counts = np.maximum(
        counts,
        1
    )

    frequency = (
        counts /
        counts.sum()
    )

    weights = (
        1.0 /
        np.sqrt(frequency)
    )

    weights = (
        weights /
        weights.mean()
    )

    print("\nClass statistics:")

    for i in range(7):

        print(
            f"  {i}: "
            f"{CLASS_NAMES[i]:<20} "
            f"{int(counts[i]):,} "
            f"weight={weights[i]:.3f}"
        )

    return weights.astype(
        np.float32
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 75)
    print(
        "REAL nuScenes -> "
        "PointNet++ 7-CLASS TRAINING"
    )
    print("=" * 75)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    # ========================================================
    # DATASET
    # ========================================================

    dataset = NuScenesDataset(
        num_points=4096,
        seed=42
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    n_total = len(dataset)

    n_val = max(
        1,
        int(n_total * 0.20)
    )

    n_train = (
        n_total -
        n_val
    )

    generator = torch.Generator().manual_seed(
        42
    )

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )

    print(
        f"\nDataset split:"
        f"\n  Training frames   : {n_train}"
        f"\n  Validation frames : {n_val}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = calculate_class_weights(
        dataset
    )

    class_weights_tensor = (
        torch.from_numpy(
            class_weights
        ).to(device)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nCreating PointNet++ model..."
    )

    model = PointNetPPClassifier(
        in_channels=4,
        num_classes=7
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    EPOCHS = 10

    best_accuracy = -1.0

    best_state = None

    history = []

    print(
        "\nStarting training..."
    )

    print(
        "Original checkpoint will NOT be modified."
    )

    print()

    start_time = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        total_correct = 0

        total_points = 0

        epoch_start = time.time()

        for batch_idx, (
            points,
            labels
        ) in enumerate(
            train_loader,
            start=1
        ):

            # (B,N,4)
            # ->
            # (B,4,N)

            points = points.transpose(
                1,
                2
            ).to(device)

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                points
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0
            )

            optimizer.step()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += int(
                (
                    predictions ==
                    labels
                ).sum()
            )

            total_points += labels.numel()

            total_loss += loss.item()

            # Show progress every batch
            print(
                f"\rEpoch {epoch}/{EPOCHS} "
                f"| batch {batch_idx}/{len(train_loader)} "
                f"| loss={loss.item():.4f}",
                end=""
            )

        train_accuracy = (
            total_correct /
            max(
                total_points,
                1
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0

        val_points = 0

        val_loss_total = 0.0

        with torch.no_grad():

            for points, labels in val_loader:

                points = points.transpose(
                    1,
                    2
                ).to(device)

                labels = labels.to(
                    device
                )

                logits = model(
                    points
                )

                loss = criterion(
                    logits,
                    labels
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                val_correct += int(
                    (
                        predictions ==
                        labels
                    ).sum()
                )

                val_points += labels.numel()

                val_loss_total += loss.item()

        val_accuracy = (
            val_correct /
            max(
                val_points,
                1
            )
        )

        val_loss = (
            val_loss_total /
            max(
                len(val_loader),
                1
            )
        )

        scheduler.step(
            val_accuracy
        )

        elapsed_epoch = (
            time.time() -
            epoch_start
        )

        print()

        print(
            f"Epoch {epoch:2d}/{EPOCHS} "
            f"| train_loss="
            f"{total_loss / max(len(train_loader),1):.4f} "
            f"| train_acc="
            f"{train_accuracy*100:.2f}% "
            f"| val_loss="
            f"{val_loss:.4f} "
            f"| val_acc="
            f"{val_accuracy*100:.2f}% "
            f"| time="
            f"{elapsed_epoch:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": (
                total_loss /
                max(
                    len(train_loader),
                    1
                )
            ),
            "val_loss": val_loss
        })

        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            print(
                f"  >>> NEW BEST: "
                f"{best_accuracy*100:.2f}%"
            )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(
        best_state
    )

    total_time = (
        time.time() -
        start_time
    )

    # ========================================================
    # SAVE NEW CHECKPOINT
    # ========================================================

    torch.save(
        model.state_dict(),
        OUTPUT_CHECKPOINT
    )

    report = {
        "dataset": "nuScenes lidarseg mini",
        "frames_total": n_total,
        "frames_training": n_train,
        "frames_validation": n_val,
        "points_per_frame": 4096,
        "epochs": EPOCHS,
        "best_validation_accuracy": best_accuracy,
        "training_time_seconds": total_time,
        "history": history
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)

    print(
        f"Best validation accuracy : "
        f"{best_accuracy*100:.2f}%"
    )

    print(
        f"Training time            : "
        f"{total_time/60:.1f} minutes"
    )

    print()

    print("NEW checkpoint:")
    print(
        OUTPUT_CHECKPOINT
    )

    print()

    print("Training report:")
    print(
        OUTPUT_REPORT
    )

    print()

    print("IMPORTANT:")
    print(
        "Your original "
        "pointnet_pp_7class.pt "
        "was NOT overwritten."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()