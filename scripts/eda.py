import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_examine(anno_path, pred_path):
    """Load and examine dataset structure."""
    anno = pd.read_csv(anno_path)
    pred = pd.read_csv(pred_path)
    
    print(f"Annotations: {anno.shape}, Predictions: {pred.shape}")
    print(f"Annotation columns: {list(anno.columns)}")
    print(f"Prediction columns: {list(pred.columns)}")
    
    return anno, pred


def check_quality(anno, pred):
    """Check data quality and handle missing values."""
    print("\nMissing values - Annotations:")
    print(anno.isnull().sum())
    
    print("\nMissing values - Predictions:")
    print(pred.isnull().sum())
    
    # Fill numeric missing with median, categorical with mode
    for df in [anno, pred]:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_val, inplace=True)
    
    print(f"Unique image IDs - Anno: {anno['image_id'].nunique()}, Pred: {pred['image_id'].nunique()}")
    return anno, pred


def compare_datasets(anno, pred):
    """Compare annotations and predictions."""
    anno_images = set(anno['image_id'])
    pred_images = set(pred['image_id'])
    
    print(f"\nCommon images: {len(anno_images & pred_images)}")
    print(f"Only in annotations: {len(anno_images - pred_images)}")
    print(f"Only in predictions: {len(pred_images - anno_images)}")
    
    anno_per_image = anno.groupby('image_id').size()
    pred_per_image = pred.groupby('image_id').size()
    
    print(f"\nDetections per image - Anno: mean={anno_per_image.mean():.1f}, Pred: mean={pred_per_image.mean():.1f}")
    
    if 'defect_class_id' in anno.columns:
        print(f"\nAnnotation classes: {anno['defect_class_id'].nunique()}")
        print("Class distribution:")
        print(anno['defect_class_id'].value_counts().head())


def analyze_confidence(pred):
    """Analyze confidence score distribution."""
    if 'confidence' not in pred.columns:
        return
    
    conf = pred['confidence']
    print(f"\nConfidence stats - Mean: {conf.mean():.3f}, Std: {conf.std():.3f}")
    print(f"Percentiles - 25th: {conf.quantile(0.25):.3f}, 50th: {conf.median():.3f}, 75th: {conf.quantile(0.75):.3f}")
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        above = (conf >= t).sum()
        print(f"Predictions ≥ {t}: {above} ({above/len(conf)*100:.1f}%)")


def visualize_summary(anno, pred):
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Confidence distribution
    if 'confidence' in pred.columns:
        axes[0,0].hist(pred['confidence'], bins=30, alpha=0.7)
        axes[0,0].axvline(0.5, color='r', linestyle='--')
        axes[0,0].set_title('Confidence Distribution')
        axes[0,0].set_xlabel('Confidence')
    
    # Detections per image
    anno_counts = anno.groupby('image_id').size()
    pred_counts = pred.groupby('image_id').size()
    
    axes[0,1].hist(anno_counts, bins=20, alpha=0.5, label='Annotations')
    axes[0,1].hist(pred_counts, bins=20, alpha=0.5, label='Predictions')
    axes[0,1].set_title('Detections per Image')
    axes[0,1].legend()
    
    # Image overlap
    anno_set = set(anno['image_id'])
    pred_set = set(pred['image_id'])
    overlap_data = [len(anno_set - pred_set), len(anno_set & pred_set), len(pred_set - anno_set)]
    axes[0,2].bar(['Anno Only', 'Both', 'Pred Only'], overlap_data)
    axes[0,2].set_title('Image Overlap')
    
    # Class distribution (annotations)
    if 'defect_class_id' in anno.columns:
        class_counts = anno['defect_class_id'].value_counts()
        axes[1,0].bar(range(len(class_counts)), class_counts.values)
        axes[1,0].set_title('Annotation Classes')
        axes[1,0].set_xlabel('Class ID')
        axes[1,0].set_ylabel('Count')
    
    # Class distribution (predictions)
    if 'prediction_class' in pred.columns:
        pred_counts = pred['prediction_class'].value_counts()
        axes[1,1].bar(range(len(pred_counts)), pred_counts.values, color='orange')
        axes[1,1].set_title('Prediction Classes')
        axes[1,1].set_xlabel('Class')
    
    # Confidence thresholds
    if 'confidence' in pred.columns:
        thresholds = np.linspace(0, 1, 51)
        percentages = [(pred['confidence'] >= t).sum() / len(pred) * 100 for t in thresholds]
        axes[1,2].plot(thresholds, percentages)
        axes[1,2].set_title('Predictions vs Threshold')
        axes[1,2].set_xlabel('Confidence Threshold')
        axes[1,2].set_ylabel('% Predictions')
    
    plt.tight_layout()
    plt.show()


def generate_report(anno, pred):
    """Generate concise analysis report."""
    report = {
        'total_annotations': len(anno),
        'total_predictions': len(pred),
        'unique_images_anno': anno['image_id'].nunique(),
        'unique_images_pred': pred['image_id'].nunique(),
        'avg_detections_anno': anno.groupby('image_id').size().mean(),
        'avg_detections_pred': pred.groupby('image_id').size().mean()
    }
    
    if 'confidence' in pred.columns:
        report['confidence_mean'] = pred['confidence'].mean()
        report['predictions_above_05'] = (pred['confidence'] >= 0.5).sum()
    
    print("\n" + "="*40)
    print("ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total: {report['total_annotations']} annotations, {report['total_predictions']} predictions")
    print(f"Images: {report['unique_images_anno']} annotated, {report['unique_images_pred']} predicted")
    print(f"Detections/image: {report['avg_detections_anno']:.1f} (anno), {report['avg_detections_pred']:.1f} (pred)")
    
    if 'confidence_mean' in report:
        print(f"Mean confidence: {report['confidence_mean']:.3f}")
        print(f"Predictions ≥ 0.5: {report['predictions_above_05']} ({report['predictions_above_05']/report['total_predictions']*100:.1f}%)")
    
    return report


def run_eda(anno_path, pred_path, visualize=True):
    """Run complete EDA pipeline."""
    print("EDA Pipeline")
    print("-" * 40)
    
    # Load and examine
    anno, pred = load_and_examine(anno_path, pred_path)
    
    # Quality check
    anno, pred = check_quality(anno, pred)
    
    # Compare datasets
    compare_datasets(anno, pred)
    
    # Analyze confidence
    analyze_confidence(pred)
    
    # Visualize
    if visualize:
        visualize_summary(anno, pred)
    
    # Generate report
    report = generate_report(anno, pred)
    
    return anno, pred, report