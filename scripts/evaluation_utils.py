import pandas as pd
from scripts.matching_utils import precompute_iou_matrix, match_predictions_one_to_one

def evaluate_at_threshold_simple(predictions, annotations, threshold, iou_threshold=0.5):
    """
    Evaluate precision, recall, F1 at a given confidence threshold (simple version)
    """
    preds_filt = predictions[predictions["confidence"] >= threshold]
    matches, gt_used, pred_used = match_predictions_one_to_one(annotations, preds_filt, iou_threshold)

    TP = len(matches)
    FP = len(preds_filt) - TP
    FN = len(annotations) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1

def evaluate_at_threshold_fast(iou_df, predictions, annotations, threshold):
    """
    Fast evaluation using precomputed IoU matrix
    """
    df = iou_df[iou_df["confidence"] >= threshold]
    df = df.sort_values("confidence", ascending=False)

    gt_used = set()
    pred_used = set()
    TP = 0

    for _, row in df.iterrows():
        if row["pred_idx"] not in pred_used and row["gt_idx"] not in gt_used:
            pred_used.add(row["pred_idx"])
            gt_used.add(row["gt_idx"])
            TP += 1

    total_preds = len(predictions[predictions["confidence"] >= threshold])
    total_gt = len(annotations)

    FP = total_preds - TP
    FN = total_gt - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def generate_threshold_metrics(predictions, annotations, n_thresholds=100, iou_threshold=0.5, fast=True):
    """
    Generate precision, recall, F1 metrics across thresholds
    """
    thresholds = [i/n_thresholds for i in range(1, n_thresholds)]
    results = []
    
    if fast:
        # Precompute IoU matrix for faster evaluation
        iou_df = precompute_iou_matrix(predictions, annotations, iou_threshold)
        for t in thresholds:
            p, r, f1 = evaluate_at_threshold_fast(iou_df, predictions, annotations, t)
            results.append((t, p, r, f1))
    else:
        for t in thresholds:
            p, r, f1 = evaluate_at_threshold_simple(predictions, annotations, t, iou_threshold)
            results.append((t, p, r, f1))
    
    metrics_df = pd.DataFrame(
        results, columns=["threshold", "precision", "recall", "f1"]
    )
    
    return metrics_df

def find_optimal_threshold(metrics_df, metric='f1'):
    """
    Find optimal threshold based on specified metric
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric {metric} not found in DataFrame")
    
    best_row = metrics_df.loc[metrics_df[metric].idxmax()]
    return best_row