import os
import glob
import numpy as np

from data_loader import LidarFrame, SemanticClass
from segmentation_engine import PointNetPPSegmentationEngine


IITH_ROOT = r"C:\Users\Acer\Downloads\IITH_LiDAR_ground_dataset_labelled_raw (2)\IITH_LiDAR_ground_dataset_labelled_raw"
LABEL_DIR = os.path.join(IITH_ROOT, "labelled_data")


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
                rgb_value = float(parts[3])

                r, g, b = decode_rgb_uint32(rgb_value)

                points.append([x, y, z])

                # IITH dataset:
                # RED = ground
                # GREEN = non-ground
                if r > g and r > b:
                    labels.append(int(SemanticClass.GROUND))
                else:
                    labels.append(int(SemanticClass.STATIC_STRUCTURE))

            except ValueError:
                continue

    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)

    intensities = np.full(len(points), 0.5, dtype=np.float32)

    return LidarFrame(
        points=points,
        labels=labels,
        intensities=intensities
    )


def evaluate():

    print("=" * 70)
    print("IITH LiDAR - POINTNET++ PROPER EVALUATION")
    print("=" * 70)

    files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.pcd")))

    print("Dataset:", LABEL_DIR)
    print("Files found:", len(files))
    print()

    if not files:
        print("ERROR: No PCD files found.")
        return

    print("[*] Loading PointNet++ model...")

    segmenter = PointNetPPSegmentationEngine()

    print("Using trained model:", segmenter.is_using_trained_model)
    print("Checkpoint:", segmenter.checkpoint_path)
    print()

    if not segmenter.is_using_trained_model:
        print("ERROR: PointNet++ checkpoint was not loaded.")
        return

    total_correct = 0
    total_points = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for path in files:

        print("-" * 70)
        print("Testing:", os.path.basename(path))

        frame = read_iith_pcd(path)

        result = segmenter.infer(frame)
        predicted = result.labels
        truth = frame.labels

        # Ground = Class 1
        truth_ground = truth == int(SemanticClass.GROUND)
        pred_ground = predicted == int(SemanticClass.GROUND)

        correct = np.sum(predicted == truth)

        tp = np.sum(truth_ground & pred_ground)
        fp = np.sum(~truth_ground & pred_ground)
        fn = np.sum(truth_ground & ~pred_ground)

        total_correct += int(correct)
        total_points += len(truth)
        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)

        accuracy = correct / len(truth)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("Points:", len(truth))
        print("Ground truth ground:", np.sum(truth_ground))
        print("Predicted ground:", np.sum(pred_ground))
        print("Accuracy:  %.2f%%" % (accuracy * 100))
        print("Precision: %.2f%%" % (precision * 100))
        print("Recall:    %.2f%%" % (recall * 100))
        print("F1 score:  %.2f%%" % (f1 * 100))

    overall_accuracy = total_correct / total_points

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    overall_f1 = (
        2 * overall_precision * overall_recall
        / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    print()
    print("=" * 70)
    print("OVERALL POINTNET++ RESULTS")
    print("=" * 70)

    print("Files tested:", len(files))
    print("Total points:", total_points)
    print("Accuracy:  %.2f%%" % (overall_accuracy * 100))
    print("Precision: %.2f%%" % (overall_precision * 100))
    print("Recall:    %.2f%%" % (overall_recall * 100))
    print("F1 score:  %.2f%%" % (overall_f1 * 100))

    print()
    print("IITH label interpretation:")
    print("RED   = Ground")
    print("GREEN = Non-ground")
    print()
    print("This evaluates ground vs non-ground only.")
    print("It does NOT validate vehicle/pedestrian/pole classification.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()