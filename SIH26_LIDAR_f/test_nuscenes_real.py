"""
Final REAL nuScenes LiDAR evaluation.

Uses:
    checkpoints/pointnet_pp_7class_nuscenes.pt

Evaluates the complete selected nuScenes frame instead of only 8192 points.
"""

import os
import json
import numpy as np
import torch

from segmentation_engine import PointNetPPClassifier


# ============================================================
# PATHS
# ============================================================

NUSCENES = r"C:\Users\Acer\Downloads\nuscenes_raw"
LIDARSEG = r"C:\Users\Acer\Downloads\nuScenes-lidarseg-mini-v1.0\extracted\lidarseg\v1.0-mini"

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "pointnet_pp_7class_nuscenes.pt"
)

SAMPLE_DATA = os.path.join(
    NUSCENES,
    "v1.0-mini",
    "sample_data.json"
)
LIDARSEG_JSON = r"C:\Users\Acer\Downloads\nuScenes-lidarseg-mini-v1.0\extracted\v1.0-mini\lidarseg.json"


# ============================================================
# CLASS DEFINITIONS
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


# nuScenes lidarseg -> project 7 classes

NUSCENES_TO_7 = {
    0: 0,

    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 6,
    6: 6,
    7: 6,
    8: 6,

    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,

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

    24: 1,
    25: 1,
    26: 1,
    27: 1,

    28: 2,
    29: 2,

    30: 4,

    31: 2,
}


# ============================================================
# FIND REAL LABELED FRAME
# ============================================================

def load_frame():

    print("Loading nuScenes metadata...")

    sample_data_path = os.path.join(
        NUSCENES,
        "v1.0-mini",
        "sample_data.json"
    )

    lidarseg_json_path = LIDARSEG_JSON
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

    # Find first labeled LIDAR_TOP keyframe
    record = None

    for x in sample_data:

        if (
            "LIDAR_TOP" in x["filename"]
            and x["is_key_frame"]
            and x["token"] in label_map
        ):
            record = x
            break

    if record is None:
        raise RuntimeError(
            "Could not find a labeled LIDAR_TOP keyframe."
        )

    # --------------------------------------------------------
    # LiDAR file
    # --------------------------------------------------------

    lidar_file = os.path.join(
        NUSCENES,
        record["filename"].replace("/", os.sep)
    )

    # --------------------------------------------------------
    # Label file
    # --------------------------------------------------------

    label_relative = label_map[record["token"]]

    label_relative = label_relative.replace(
        "/",
        os.sep
    )

    # The lidarseg.json filename is normally:
    #
    # lidarseg/v1.0-mini/XXXX_lidarseg.bin
    #
    # Our LIDARSEG variable already points to:
    #
    # extracted/lidarseg/v1.0-mini
    #
    # Therefore remove that prefix first.

    prefix = (
        "lidarseg"
        + os.sep
        + "v1.0-mini"
        + os.sep
    )

    if label_relative.lower().startswith(
        prefix.lower()
    ):
        label_relative = label_relative[
            len(prefix):
        ]

    label_file = os.path.join(
        LIDARSEG,
        label_relative
    )

    print()
    print("LiDAR file:")
    print(lidar_file)

    print()
    print("Label file:")
    print(label_file)

    # --------------------------------------------------------
    # Verify files exist
    # --------------------------------------------------------

    if not os.path.isfile(lidar_file):

        raise FileNotFoundError(
            f"LiDAR file not found:\n{lidar_file}"
        )

    if not os.path.isfile(label_file):

        raise FileNotFoundError(
            f"Label file not found:\n{label_file}"
        )

    # --------------------------------------------------------
    # Load LiDAR
    # --------------------------------------------------------

    points = np.fromfile(
        lidar_file,
        dtype=np.float32
    ).reshape(-1, 5)

    # --------------------------------------------------------
    # Load lidarseg labels
    # --------------------------------------------------------

    raw_labels = np.fromfile(
        label_file,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Check alignment
    # --------------------------------------------------------

    if len(points) != len(raw_labels):

        raise RuntimeError(
            f"Point/label mismatch: "
            f"{len(points)} vs {len(raw_labels)}"
        )

    return (
        points,
        raw_labels,
        lidar_file,
        label_file
    )
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FINAL REAL nuScenes LiDAR + PointNet++ EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Load real frame
    # --------------------------------------------------------

    (
        points,
        raw_labels,
        lidar_file,
        label_file
    ) = load_frame()

    print()
    print("LiDAR file:")
    print(lidar_file)

    print()
    print("Label file:")
    print(label_file)

    print()
    print("Point cloud shape:", points.shape)
    print("Raw label shape:", raw_labels.shape)

    print("Point/label alignment: OK")

    # --------------------------------------------------------
    # Map labels
    # --------------------------------------------------------

    gt = np.array(
        [
            NUSCENES_TO_7.get(
                int(x),
                0
            )
            for x in raw_labels
        ],
        dtype=np.int64
    )

    print()
    print("GROUND TRUTH DISTRIBUTION")
    print("-" * 50)

    for i, name in enumerate(CLASS_NAMES):

        count = int(
            np.sum(gt == i)
        )

        percentage = (
            count / len(gt) * 100
        )

        print(
            f"{i}: {name:<22} "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading trained checkpoint:")
    print(CHECKPOINT)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = PointNetPPClassifier(
        in_channels=4,
        num_classes=7
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    # --------------------------------------------------------
    # Prepare ALL points
    # --------------------------------------------------------

    xyzi = points[:, :4]

    total_points = len(xyzi)

    print()
    print(
        f"Running inference on ALL "
        f"{total_points:,} real LiDAR points..."
    )

    tensor = torch.from_numpy(
        xyzi.T
    ).float().unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(tensor)

        predictions = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy().reshape(-1)

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    accuracy = float(
        np.mean(
            predictions == gt
        )
    )
    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = np.zeros((7, 7), dtype=np.int64)

    for true_label, predicted_label in zip(gt, predictions):
        cm[int(true_label), int(predicted_label)] += 1

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL REAL nuScenes RESULT")
    print("=" * 75)

    print(
        f"Points evaluated : {total_points:,}"
    )

    print(
        f"Accuracy         : {accuracy * 100:.2f}%"
    )

    print()
    print("PER-CLASS METRICS")
    print("-" * 75)

    for i, name in enumerate(CLASS_NAMES):

        true_positive = cm[i, i]

        actual = cm[i, :].sum()

        predicted = cm[:, i].sum()

        if actual > 0:
            recall = (
                true_positive / actual
            )
        else:
            recall = 0.0

        if predicted > 0:
            precision = (
                true_positive / predicted
            )
        else:
            precision = 0.0

        if (
            precision + recall
            > 0
        ):
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        union = (
            actual
            + predicted
            - true_positive
        )

        if union > 0:
            iou = (
                true_positive
                / union
            )
        else:
            iou = 0.0

        print(
            f"{i}: {name:<22} "
            f"Precision={precision*100:6.2f}% "
            f"Recall={recall*100:6.2f}% "
            f"F1={f1*100:6.2f}% "
            f"IoU={iou*100:6.2f}%"
        )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print()
    print("PREDICTED DISTRIBUTION")
    print("-" * 50)

    for i, name in enumerate(CLASS_NAMES):

        count = int(
            np.sum(
                predictions == i
            )
        )

        percentage = (
            count / total_points * 100
        )

        print(
            f"{i}: {name:<22} "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("CONFUSION MATRIX")
    print("-" * 75)

    print(
        "Rows = Ground Truth"
    )

    print(
        "Columns = Prediction"
    )

    print()

    header = "          "

    for i in range(7):
        header += f"{i:>9}"

    print(header)

    for i in range(7):

        row = f"{i:<9}"

        for j in range(7):

            row += f"{cm[i,j]:>9,}"

        print(row)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "checkpoints",
        "final_nuscenes_evaluation.json"
    )

    result = {
        "checkpoint": CHECKPOINT,
        "lidar_file": lidar_file,
        "label_file": label_file,
        "points_evaluated": total_points,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "ground_truth_distribution": {
            CLASS_NAMES[i]: int(
                np.sum(gt == i)
            )
            for i in range(7)
        },
        "prediction_distribution": {
            CLASS_NAMES[i]: int(
                np.sum(
                    predictions == i
                )
            )
            for i in range(7)
        },
        "confusion_matrix": cm.tolist()
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print()
    print("Evaluation report saved to:")
    print(output_file)

    print()
    print(
        "Trained checkpoint used:"
    )

    print(CHECKPOINT)

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()