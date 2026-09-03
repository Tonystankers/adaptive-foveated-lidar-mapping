import os
import glob
import numpy as np
from data_loader import LidarFrame, SemanticClass
from segmentation_engine import GeometricSegmentationEngine

IITH_ROOT = r"C:\Users\Acer\Downloads\IITH_LiDAR_ground_dataset_labelled_raw (2)\IITH_LiDAR_ground_dataset_labelled_raw"
LABEL_DIR = os.path.join(IITH_ROOT, "labelled_data")


def decode_rgb_uint32(rgb_value):
    """Decode packed PCD RGB value into R, G, B."""
    rgb = int(rgb_value)
    r = (rgb >> 16) & 255
    g = (rgb >> 8) & 255
    b = rgb & 255
    return r, g, b


def read_iith_pcd(path):
    """Read an ASCII IITH PCD containing XYZ + packed RGB."""
    points = []
    ground_truth = []

    with open(path, "r", encoding="utf-8") as f:
        data_started = False

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.upper() == "DATA ASCII":
                data_started = True
                continue

            if not data_started:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            x, y, z = map(float, parts[:3])
            rgb = int(parts[3])

            r, g, b = decode_rgb_uint32(rgb)

            points.append([x, y, z])

            # IITH ground truth:
            # red = ground
            # green = non-ground
            if r == 255 and g == 0 and b == 0:
                ground_truth.append(int(SemanticClass.GROUND))
            else:
                ground_truth.append(int(SemanticClass.STATIC_STRUCTURE))

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(ground_truth, dtype=np.uint8),
    )


def main():
    files = sorted(glob.glob(os.path.join(LABEL_DIR, "*_labelled.pcd")))

    print("=" * 70)
    print("IITH LiDAR DATASET TEST")
    print("=" * 70)
    print("Dataset:", LABEL_DIR)
    print("Files found:", len(files))
    print()

    if not files:
        print("ERROR: No labelled PCD files found.")
        return

    segmenter = GeometricSegmentationEngine()

    total_points = 0
    total_correct = 0

    all_true = []
    all_pred = []

    for path in files:
        print("-" * 70)
        print("Testing:", os.path.basename(path))

        points, true_labels = read_iith_pcd(path)

        intensities = np.full(
            len(points),
            0.5,
            dtype=np.float32
        )

        frame = LidarFrame(
            points=points,
            labels=np.zeros(len(points), dtype=np.uint8),
            intensities=intensities,
        )

        result = segmenter.infer(frame)
        predicted = result.labels

        # Convert your 7-class prediction to the same
        # binary Ground / Non-ground evaluation used by IITH.
        pred_ground = predicted == int(SemanticClass.GROUND)
        true_ground = true_labels == int(SemanticClass.GROUND)

        correct = int(np.sum(pred_ground == true_ground))
        accuracy = correct / len(points) if len(points) else 0.0

        tp = int(np.sum(pred_ground & true_ground))
        tn = int(np.sum((~pred_ground) & (~true_ground)))
        fp = int(np.sum(pred_ground & (~true_ground)))
        fn = int(np.sum((~pred_ground) & true_ground))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        print("Points:", len(points))
        print("Ground points:", int(np.sum(true_ground)))
        print("Non-ground points:", int(np.sum(~true_ground)))
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 score:  {f1 * 100:.2f}%")

        total_points += len(points)
        total_correct += correct

        all_true.extend(true_ground.tolist())
        all_pred.extend(pred_ground.tolist())

    all_true = np.asarray(all_true, dtype=bool)
    all_pred = np.asarray(all_pred, dtype=bool)

    tp = int(np.sum(all_pred & all_true))
    tn = int(np.sum((~all_pred) & (~all_true)))
    fp = int(np.sum(all_pred & (~all_true)))
    fn = int(np.sum((~all_pred) & all_true))

    accuracy = (tp + tn) / len(all_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print()
    print("=" * 70)
    print("OVERALL IITH RESULTS")
    print("=" * 70)
    print("Files tested:", len(files))
    print("Total points:", total_points)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 score:  {f1 * 100:.2f}%")
    print()
    print("This evaluates ONLY ground vs non-ground.")
    print("It does NOT validate vehicle/pedestrian/pole classification.")
    print("=" * 70)


if __name__ == "__main__":
    main()