import numpy as np
# Import necessary functions for instance matching evaluation.
from connectomics.utils.evaluate import _check_label_array, _raise, matching_criteria, label_overlap
from skimage.segmentation import relabel_sequential

#############################################
# Evaluation Function using instance_matching
#############################################
def instance_matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """
    Calculate detection/instance segmentation metrics between ground truth and predicted label images.
    """
    _check_label_array(y_true, 'y_true')
    _check_label_array(y_pred, 'y_pred')

    if y_true.shape != y_pred.shape:
        _raise(ValueError("y_true ({}) and y_pred ({}) have different shapes".format(y_true.shape, y_pred.shape)))
    if criterion not in matching_criteria:
        _raise(ValueError("Matching criterion '%s' not supported." % criterion))

    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float, thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)
    map_rev_true = np.array(map_rev_true)
    map_rev_pred = np.array(map_rev_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # Ignore background (label 0)
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape

    tp = (scores >= thresh).sum()
    fp = n_pred - tp
    fn = n_true - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        'criterion': criterion,
        'thresh': thresh,
        'fp': fp,
        'tp': tp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        'f1': f1,
        'n_true': n_true,
        'n_pred': n_pred,
        'mean_true_score': scores.mean() if n_true > 0 else 0,
        'mean_matched_score': scores[scores >= thresh].mean() if tp > 0 else 0,
        'panoptic_quality': tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0,
    }

    if report_matches:
        matched_pairs = [(i + 1, np.argmax(scores[i]) + 1) for i in range(n_true)]
        matched_scores = [scores[i - 1, j - 1] for i, j in matched_pairs]
        matched_pairs = [(map_rev_true[i], map_rev_pred[j]) for i, j in matched_pairs]
        result.update({
            'matched_pairs': matched_pairs,
            'matched_scores': matched_scores,
        })

    return result

def evaluate_res(y_true, y_pred):
    """
    Evaluate the watershed segmentation result using instance matching.
    Both pred_file and gt_file are paths to tif files.
    """

    # Compute evaluation metrics using instance_matching.
    metrics = instance_matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        # Only print numerical metrics
        if isinstance(value, (int, float, np.float32, np.float64)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save metrics as a text file next to the prediction.
    try:
        txt_file = pred_file.replace('.tif', '.txt')
        with open(txt_file, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
    except Exception as e:
        print(f"Failed to save metrics to txt: {e}")
    
    return metrics