import os
import glob
import numpy as np
import torch

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

                # IITH labels
                # RED   = Ground
                # GREEN = Non-ground

                if r > g and r > b:
                    labels.append(0)
                else:
                    labels.append(1)

            except ValueError:
                continue

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


print("=" * 70)
print("IITH LiDAR - 2 CLASS POINTNET++ EVALUATION")
print("=" * 70)

files = sorted(
    glob.glob(os.path.join(LABEL_DIR, "*.pcd"))
)

print("Files found:", len(files))

print()
print("Loading trained IITH model...")

model = PointNetPPClassifier(
    in_channels=4,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location="cpu"
    )
)

model.eval()

print("Checkpoint:", CHECKPOINT)

print()
print("-" * 70)


total_points = 0
total_correct = 0

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0


for path in files:

    points, labels = read_iith_pcd(path)

    print()
    print("Testing:", os.path.basename(path))
    print("Points:", len(points))

    # Same normalization used during training
    center = points.mean(axis=0)
    points_norm = points - center

    scale = np.max(
        np.linalg.norm(points_norm, axis=1)
    )

    if scale > 0:
        points_norm = points_norm / scale

    intensity = np.full(
        (len(points_norm), 1),
        0.5,
        dtype=np.float32
    )

    xyzi = np.concatenate(
        [points_norm, intensity],
        axis=1
    ).astype(np.float32)

    tensor = torch.from_numpy(
        xyzi
    ).unsqueeze(0).transpose(1, 2)

    with torch.no_grad():

        logits = model(tensor)

        predictions = torch.argmax(
            logits,
            dim=1
        ).squeeze(0).numpy()

    # Ground truth
    gt_ground = np.sum(labels == 0)
    gt_non_ground = np.sum(labels == 1)

    pred_ground = np.sum(predictions == 0)
    pred_non_ground = np.sum(predictions == 1)

    correct = np.sum(
        predictions == labels
    )

    tp = np.sum(
        (predictions == 1) &
        (labels == 1)
    )

    fp = np.sum(
        (predictions == 1) &
        (labels == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (labels == 1)
    )

    tn = np.sum(
        (predictions == 0) &
        (labels == 0)
    )

    accuracy = correct / len(labels)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    print("Ground truth ground:    ", gt_ground)
    print("Ground truth non-ground:", gt_non_ground)

    print("Predicted ground:       ", pred_ground)
    print("Predicted non-ground:   ", pred_non_ground)

    print()
    print("Accuracy: ", f"{accuracy * 100:.2f}%")
    print("Precision:", f"{precision * 100:.2f}%")
    print("Recall:   ", f"{recall * 100:.2f}%")
    print("F1 score: ", f"{f1 * 100:.2f}%")

    total_points += len(labels)
    total_correct += correct

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

    print("-" * 70)


overall_accuracy = total_correct / total_points

overall_precision = total_tp / max(
    total_tp + total_fp,
    1
)

overall_recall = total_tp / max(
    total_tp + total_fn,
    1
)

overall_f1 = (
    2 * overall_precision * overall_recall /
    max(
        overall_precision + overall_recall,
        1e-8
    )
)


print()
print("=" * 70)
print("OVERALL IITH POINTNET++ RESULTS")
print("=" * 70)

print("Total points:", total_points)

print()
print("Accuracy: ", f"{overall_accuracy * 100:.2f}%")
print("Precision:", f"{overall_precision * 100:.2f}%")
print("Recall:   ", f"{overall_recall * 100:.2f}%")
print("F1 score: ", f"{overall_f1 * 100:.2f}%")

print()
print("Confusion Matrix")
print("----------------")

print("True Ground / Pred Ground:",
      total_tn)

print("True Ground / Pred Non-ground:",
      total_fp)

print("True Non-ground / Pred Ground:",
      total_fn)

print("True Non-ground / Pred Non-ground:",
      total_tp)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)