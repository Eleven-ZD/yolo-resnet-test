# ===========================
# 评估指标
# ===========================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(labels, preds):
    """计算并返回常用指标"""
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def print_metrics(metrics, prefix="验证集"):
    """格式化打印指标"""
    print(f"\n{'─' * 40}")
    print(f"  {prefix}")
    print(f"  Accuracy :  {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall   :  {metrics['recall']:.4f}")
    print(f"  F1-score :  {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"             预测 cap  预测 no_cap")
    print(f"    真 cap      {cm[0][0]:<6}    {cm[0][1]:<6}")
    print(f"  真 no_cap     {cm[1][0]:<6}    {cm[1][1]:<6}")
    print(f"{'─' * 40}")
