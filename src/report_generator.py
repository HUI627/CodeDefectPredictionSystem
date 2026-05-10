"""
报告生成模块
"""
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    """实验报告生成器"""

    def __init__(self, config):
        self.config = config
        self.reports_dir = config.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, metrics):
        """生成实验报告"""
        logger.info("生成实验报告...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"experiment_report_{timestamp}.md"

        report_content = self._create_report_content(metrics)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✓ 报告已生成: {report_path}")
        return report_path

    def _create_report_content(self, metrics):
        """创建报告内容"""
        report = f"""# 代码缺陷预测系统 - 实验报告

## 1. 实验概述

**实验时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**模型**: CodeBERT-based Defect Predictor
**任务**: 二分类（有缺陷/无缺陷）

## 2. 实验配置

### 2.1 模型配置
- 预训练模型: {self.config.model_name}
- 最大序列长度: {self.config.max_seq_length}
- Dropout率: {self.config.dropout_rate}

### 2.2 训练配置
- Batch Size: {self.config.batch_size}
- 学习率: {self.config.learning_rate}
- 训练轮数: {self.config.num_epochs}
- 早停耐心值: {self.config.early_stopping_patience}
- 优化器: AdamW
- 设备: {self.config.device}

## 3. 评估结果

### 3.1 性能指标

| 指标 | 数值 |
|------|------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |

### 3.2 混淆矩阵

```
真实标签 \\ 预测标签    无缺陷    有缺陷
无缺陷                {metrics['confusion_matrix'][0][0]:6d}    {metrics['confusion_matrix'][0][1]:6d}
有缺陷                {metrics['confusion_matrix'][1][0]:6d}    {metrics['confusion_matrix'][1][1]:6d}
```

## 4. 结果分析

### 4.1 模型性能
- 模型在测试集上的准确率为 **{metrics['accuracy']:.2%}**
- F1分数为 **{metrics['f1']:.4f}**，表明模型在精确率和召回率之间取得了良好的平衡
- AUC-ROC为 **{metrics['auc_roc']:.4f}**，说明模型具有较好的分类能力

### 4.2 优势
1. **深度学习方法**: 使用CodeBERT预训练模型，能够自动学习代码的语义特征
2. **端到端学习**: 无需手工设计特征，模型自动从代码中提取有用信息
3. **迁移学习**: 利用大规模代码语料库预训练的知识，提升性能

### 4.3 改进方向
1. **数据增强**: 使用更多的代码缺陷数据集进行训练
2. **模型集成**: 结合多个模型的预测结果，提升整体性能
3. **注意力可视化**: 分析模型关注的代码片段，提升可解释性

## 5. 可视化结果

实验生成了以下可视化图表：
- 训练历史曲线: `results/figures/training_history.png`
- 混淆矩阵: `results/figures/confusion_matrix.png`
- ROC曲线: `results/figures/roc_curve.png`
- Precision-Recall曲线: `results/figures/precision_recall_curve.png`
- 类别分布: `results/figures/class_distribution.png`

## 6. 结论

本实验成功构建了基于CodeBERT的代码缺陷预测系统，在测试集上取得了良好的性能。
模型能够有效识别代码中的潜在缺陷，为代码质量保证提供了有力支持。

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report
