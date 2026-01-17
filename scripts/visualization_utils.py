import matplotlib.pyplot as plt
import seaborn as sns

def plot_threshold_metrics(metrics_df, optimal_threshold=None):
    """
    Plot precision, recall, and F1 vs threshold
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(metrics_df["threshold"], metrics_df["precision"], 
             label="Precision", linewidth=2)
    plt.plot(metrics_df["threshold"], metrics_df["recall"], 
             label="Recall", linewidth=2)
    plt.plot(metrics_df["threshold"], metrics_df["f1"], 
             label="F1-score", linewidth=2)
    
    if optimal_threshold is not None:
        plt.axvline(optimal_threshold, linestyle="--", 
                   color='red', label=f"Optimal threshold = {optimal_threshold:.3f}")
    
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision / Recall / F1 vs Confidence Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_precision_recall_curve(metrics_df):
    """
    Plot precision vs recall curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df["recall"], metrics_df["precision"], linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    return plt.gcf()

def plot_confidence_distribution(predictions, annotations=None):
    """
    Plot distribution of confidence scores
    """
    plt.figure(figsize=(10, 6))
    
    # Plot predictions distribution
    plt.hist(predictions["confidence"], bins=50, alpha=0.7, 
             label="Predictions", color='blue')
    
    # If annotations available, mark them
    if annotations is not None:
        plt.axvline(x=0.5, color='red', linestyle='--', 
                   label='Default threshold (0.5)')
    
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Confidence Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()