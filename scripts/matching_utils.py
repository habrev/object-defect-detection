import pandas as pd
from scripts.polygon_utils import polygon_iou

def match_predictions_one_to_one(gt_df, pred_df, iou_threshold=0.5):
    """
    One-to-one matching using highest-confidence first strategy
    Returns: matches (list of tuples), gt_used (set), pred_used (set)
    """
    gt_used = set()
    pred_used = set()
    matches = []

    pred_sorted = pred_df.sort_values("confidence", ascending=False)

    for p_idx, p_row in pred_sorted.iterrows():
        best_iou = 0
        best_gt_idx = None

        for g_idx, g_row in gt_df.iterrows():
            if g_idx in gt_used:
                continue

            iou = polygon_iou(p_row["polygon"], g_row["polygon"])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx is not None:
            gt_used.add(best_gt_idx)
            pred_used.add(p_idx)
            matches.append((p_idx, best_gt_idx))

    return matches, gt_used, pred_used

def precompute_iou_matrix(predictions, annotations, iou_threshold=0.5):
    """
    Precompute IoU matrix for faster threshold evaluation
    Returns DataFrame with pred_idx, gt_idx, iou, confidence
    """
    iou_records = []

    for p_idx, p_row in predictions.iterrows():
        for g_idx, g_row in annotations.iterrows():
            iou = polygon_iou(p_row["polygon"], g_row["polygon"])
            if iou >= iou_threshold:
                iou_records.append({
                    "pred_idx": p_idx,
                    "gt_idx": g_idx,
                    "iou": iou,
                    "confidence": p_row["confidence"]
                })

    return pd.DataFrame(iou_records)