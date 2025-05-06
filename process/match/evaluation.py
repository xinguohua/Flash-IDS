from sklearn import metrics
from process.match.loss import *
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import numpy as np

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if config['training']['loss'] == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config['training']['loss'] == 'hamming':
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

    See `tf.metrics.auc` for more details about this metric.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1.
      labels: [n_examples] int.  Labels are either +1 or -1.
      **auc_args: other arguments that can be used by `tf.metrics.auc`.

    Returns:
      auc: the area under the ROC curve.
    """
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)


def compute_metrics(y_scores, y_true, threshold=0.5):
    """
    Compute evaluation metrics: Accuracy, F1, AUC, Precision, Recall, FPR.
    Args:
        y_true: array-like of shape (n_samples,) — Ground truth labels (0 or 1)
        y_scores: array-like of shape (n_samples,) — Model predicted scores
        threshold: float — Threshold to convert scores to binary predictions
    Returns:
        dict: Dictionary of metrics: Acc, F1, AUC, Prec, Recall, FPR
    """
    # Convert to binary predictions based on threshold
    y_true = y_true.detach().cpu().numpy()
    y_true = (y_true + 1) / 2

    y_scores = y_scores.detach().cpu().numpy()

    scores_max = np.max(y_scores)
    scores_min = np.min(y_scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores_normal = (y_scores - scores_min) / (scores_max - scores_min + 1e-8)
    y_pred = (scores_normal >= threshold).astype(int)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # Compute FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-8)

    return {
        "Acc": acc,
        "F1": f1,
        "AUC": auc,
        "Prec": prec,
        "Recall": rec,
        "FPR": fpr
    }
