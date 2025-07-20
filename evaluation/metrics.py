from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def evaluate_anomaly_detection(y_true, anomaly_scores, threshold=None):
    if threshold is None:
        threshold = sorted(anomaly_scores)[int(0.95 * len(anomaly_scores))]  # top 5% anomalies

    y_pred = [1 if score > threshold else 0 for score in anomaly_scores]

    auc_roc = roc_auc_score(y_true, anomaly_scores)
    auc_pr = average_precision_score(y_true, anomaly_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    fpr = sum((y_true == 0) & (np.array(y_pred) == 1)) / sum(y_true == 0)

    return {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "FPR": fpr
    }
