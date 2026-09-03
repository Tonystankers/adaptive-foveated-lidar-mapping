"""
evaluation.py - Classification Accuracy & Confusion Metrics
SIH26053: Adaptive Variable Resolution 2.5D LiDAR Mapping Pipeline

This module fills the previously-missing "Performance Metrics: ... high
accuracy in object classification" requirement. It provides:
    - Vectorized confusion-matrix computation
    - Per-class precision / recall / F1 + overall accuracy & macro-F1
    - Accuracy broken down by radial distance-from-sensor bucket (so you can
      see how classification quality degrades with range - directly what the
      brief asks for: "high accuracy ... across varying distances")
    - A human-readable report formatter used by main.py / train_model.py

Any segmentation engine (GeometricSegmentationEngine or
PointNetPPSegmentationEngine) can be evaluated as long as ground-truth labels
are available for the input LidarFrame (true for SyntheticLidarGenerator
output, and for SemanticKittiDataset when .label files are present).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from data_loader import LidarFrame, SemanticClass


# ==============================================================================
# 1. CONFUSION MATRIX & CLASSIFICATION METRICS
# ==============================================================================

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Vectorized confusion matrix computation via bincount on a flattened index.
    Rows = ground truth class, Columns = predicted class.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict:
    """
    Derives overall accuracy, per-class precision/recall/F1/support, and
    macro-averaged F1 from a confusion matrix.
    """
    num_classes = cm.shape[0]
    total = cm.sum()
    correct = np.trace(cm)
    overall_accuracy = float(correct / total) if total > 0 else 0.0

    per_class = {}
    f1_scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[SemanticClass.get_name(c)] = {
            "class_id": c,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_points": int(total),
        "per_class": per_class,
    }


def format_metrics_report(metrics: Dict, title: str = "CLASSIFICATION METRICS") -> str:
    """Pretty-prints a metrics dict (as returned by metrics_from_confusion_matrix)."""
    lines = []
    lines.append("-" * 80)
    lines.append(f"{title:^80}")
    lines.append("-" * 80)
    lines.append(f"  Overall Accuracy : {metrics['overall_accuracy']*100:.2f}%   "
                 f"Macro-F1: {metrics['macro_f1']*100:.2f}%   "
                 f"Total Points: {metrics['total_points']:,}")
    lines.append("-" * 80)
    lines.append(f"  {'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Support':>10}")
    lines.append("-" * 80)
    for name, m in metrics["per_class"].items():
        if m["support"] == 0:
            continue
        lines.append(
            f"  {name:<20} | {m['precision']*100:9.2f}% | {m['recall']*100:9.2f}% | "
            f"{m['f1']*100:9.2f}% | {m['support']:10,d}"
        )
    lines.append("-" * 80)
    return "\n".join(lines)


# ==============================================================================
# 2. ACCURACY-VS-DISTANCE EVALUATION
# ==============================================================================

DEFAULT_DISTANCE_BINS: List[Tuple[float, float]] = [
    (0.0, 10.0), (10.0, 40.0), (40.0, 70.0), (70.0, 100.0),
]


def evaluate_engine(
    engine,
    frames: List[LidarFrame],
    num_classes: int = None,
    distance_bins: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Runs `engine.infer(frame)` over a list of ground-truth-labeled frames and
    computes:
      - Overall confusion matrix & per-class precision/recall/F1
      - Accuracy binned by radial distance from the sensor (the core ask:
        "high accuracy in object classification across varying distances")

    `engine` can be a GeometricSegmentationEngine or PointNetPPSegmentationEngine
    (anything exposing `.infer(frame) -> LidarFrame` with `.labels` populated).
    """
    if num_classes is None:
        num_classes = SemanticClass.num_classes()
    if distance_bins is None:
        distance_bins = DEFAULT_DISTANCE_BINS

    overall_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    bin_cms = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in distance_bins]

    for frame in frames:
        gt_labels = frame.labels
        pred_frame = engine.infer(frame)
        pred_labels = pred_frame.labels

        overall_cm += compute_confusion_matrix(gt_labels, pred_labels, num_classes)

        radii = np.hypot(frame.points[:, 0], frame.points[:, 1])
        for i, (r_min, r_max) in enumerate(distance_bins):
            mask = (radii >= r_min) & (radii < r_max)
            if not np.any(mask):
                continue
            bin_cms[i] += compute_confusion_matrix(gt_labels[mask], pred_labels[mask], num_classes)

    overall_metrics = metrics_from_confusion_matrix(overall_cm)

    distance_accuracy = []
    for (r_min, r_max), cm in zip(distance_bins, bin_cms):
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else None
        distance_accuracy.append({
            "range_m": f"{r_min:.0f}-{r_max:.0f}",
            "r_min": r_min,
            "r_max": r_max,
            "accuracy": round(acc, 4) if acc is not None else None,
            "num_points": int(total),
        })

    return {
        "overall_confusion_matrix": overall_cm.tolist(),
        "overall_metrics": overall_metrics,
        "distance_accuracy": distance_accuracy,
        "num_frames_evaluated": len(frames),
    }


def format_distance_accuracy_report(eval_result: Dict, title: str = "ACCURACY vs. DISTANCE") -> str:
    lines = []
    lines.append("-" * 80)
    lines.append(f"{title:^80}")
    lines.append("-" * 80)
    lines.append(f"  {'Distance Range':<18} | {'Accuracy':>10} | {'Points Evaluated':>18}")
    lines.append("-" * 80)
    for d in eval_result["distance_accuracy"]:
        acc_str = f"{d['accuracy']*100:.2f}%" if d["accuracy"] is not None else "N/A"
        lines.append(f"  {d['range_m'] + ' m':<18} | {acc_str:>10} | {d['num_points']:18,d}")
    lines.append("-" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test: evaluate the geometric heuristic engine against synthetic ground truth
    from data_loader import SyntheticLidarGenerator
    from segmentation_engine import GeometricSegmentationEngine

    print("=" * 80)
    print("  SIH26053: evaluation.py self-test (Geometric engine vs. synthetic GT)")
    print("=" * 80)

    generator = SyntheticLidarGenerator(seed=7, max_range=100.0)
    engine = GeometricSegmentationEngine()
    frames = [generator.generate_frame(frame_id=i, timestamp=i * 0.1) for i in range(6)]

    result = evaluate_engine(engine, frames)
    print(format_metrics_report(result["overall_metrics"], title="GEOMETRIC ENGINE - OVERALL METRICS"))
    print()
    print(format_distance_accuracy_report(result, title="GEOMETRIC ENGINE - ACCURACY vs DISTANCE"))
