"""
可视化模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)

class Visualizer:
    """可视化器"""

    def __init__(self, config):
        self.config = config
        self.figures_dir = config.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_training_history(self):
        """绘制训练历史"""
        try:
            with open(self.config.metrics_dir / 'training_history.json', 'r') as f:
                history = json.load(f)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            epochs = range(1, len(history['train_losses']) + 1)
            ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
            ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            ax2.plot(epochs, history['train_accs'], 'b-', label='Train Acc')
            ax2.plot(epochs, history['val_accs'], 'r-', label='Val Acc')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(self.figures_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✓ 训练历史图表已保存")
        except Exception as e:
            logger.warning(f"无法绘制训练历史: {str(e)}")

    def plot_confusion_matrix(self, labels, predictions):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Defect', 'Defect'],
                   yticklabels=['No Defect', 'Defect'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ 混淆矩阵已保存")

    def plot_roc_curve(self, labels, probs):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ ROC曲线已保存")

    def plot_precision_recall_curve(self, labels, probs):
        """绘制Precision-Recall曲线"""
        precision, recall, _ = precision_recall_curve(labels, probs)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Precision-Recall曲线已保存")

    def plot_class_distribution(self, labels, predictions):
        """绘制类别分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        unique, counts = np.unique(labels, return_counts=True)
        ax1.bar(['No Defect', 'Defect'], counts, color=['green', 'red'], alpha=0.7)
        ax1.set_title('True Label Distribution')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)

        unique, counts = np.unique(predictions, return_counts=True)
        ax2.bar(['No Defect', 'Defect'], counts, color=['green', 'red'], alpha=0.7)
        ax2.set_title('Predicted Label Distribution')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ 类别分布图已保存")
