# src/core/evaluator.py

import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from transformers import EvalPrediction
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_metrics_factory(task_type: str, threshold: float = 0.5):
    """
    Factory function to return the appropriate compute_metrics function.
    """
    if task_type == 'single_label':
        return compute_single_label_metrics
    elif task_type == 'multi_label':
        def compute_multi_label_metrics(p: EvalPrediction) -> Dict[str, float]:
            return compute_multi_label_metrics_with_threshold(p, threshold)
        return compute_multi_label_metrics
    else:
        raise ValueError(f"Unsupported task type for metrics: {task_type}")

def compute_single_label_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for single-label classification.
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_multi_label_metrics_with_threshold(p: EvalPrediction, threshold: float) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification.
    """
    logits = p.predictions
    labels = p.label_ids

    # Apply sigmoid to logits to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))

    # Use threshold to get predictions
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1

    # Calculate metrics
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')

    # ROC-AUC needs probabilities
    try:
        roc_auc = roc_auc_score(labels, probs, average='micro')
    except ValueError:
        roc_auc = -1.0 # Handle case where all labels for a sample are the same
        logger.warning("Could not compute ROC-AUC score. This might happen in small batches.")

    return {
        f'f1_micro (threshold_{threshold})': f1_micro,
        f'f1_macro (threshold_{threshold})': f1_macro,
        'roc_auc_micro': roc_auc,
    }

