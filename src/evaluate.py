"""
模型评估模块
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import logging
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """模型评估器"""

    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, test_df):
        """评估模型"""
        logger.info("开始评估模型...")

        from src.dataset import CodeDefectDataset
        test_dataset = CodeDefectDataset(test_df, self.tokenizer, self.config.max_seq_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_predictions, all_probs)
        self._log_metrics(metrics)

        return metrics, all_predictions, all_labels, all_probs

    def _compute_metrics(self, labels, predictions, probs):
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'auc_roc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
        return metrics

    def _log_metrics(self, metrics):
        """记录评估指标"""
        logger.info("\n" + "=" * 60)
        logger.info("评估结果:")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info("=" * 60)

    def save_results(self, metrics, predictions, labels, probs):
        """保存评估结果"""
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config.metrics_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        results = {
            'predictions': [int(p) for p in predictions],
            'labels': [int(l) for l in labels],
            'probabilities': [float(p) for p in probs]
        }

        with open(self.config.metrics_dir / 'predictions.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ 评估结果已保存到 {self.config.metrics_dir}")
