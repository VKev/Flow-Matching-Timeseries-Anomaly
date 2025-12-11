import numpy as np


def binary_classification_metrics(y_true, y_pred, zero_division=0.0):
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)

    tp = np.sum(yt & yp)
    fp = np.sum(~yt & yp)
    fn = np.sum(yt & ~yp)
    tn = np.sum(~yt & ~yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return acc, precision, recall, f1, (tp, fp, fn, tn)
