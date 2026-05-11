# 代码缺陷预测系统 - 实验报告

## 1. 实验概述

**实验时间**: 2026-05-11 16:28:00
**模型**: CodeBERT-based Defect Predictor
**任务**: 二分类（有缺陷/无缺陷）

## 2. 实验配置

### 2.1 模型配置
- 预训练模型: /root/autodl-tmp/CodeDefectPredictionSystem/models/codebert-base
- 最大序列长度: 512
- Dropout率: 0.1

### 2.2 训练配置
- Batch Size: 16
- 学习率: 2e-05
- 训练轮数: 5
- 早停耐心值: 3
- 优化器: AdamW
- 设备: cuda

## 3. 评估结果

### 3.1 性能指标

| 指标 | 数值 |
|------|------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |
| AUC-ROC | 1.0000 |

### 3.2 混淆矩阵

```
真实标签 \ 预测标签    无缺陷    有缺陷
无缺陷                    57         0
有缺陷                     0        64
```

## 4. 结果分析

### 4.1 模型性能
- 模型在测试集上的准确率为 **100.00%**
- F1分数为 **1.0000**，表明模型在精确率和召回率之间取得了良好的平衡
- AUC-ROC为 **1.0000**，说明模型具有较好的分类能力

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

**报告生成时间**: 2026-05-11 16:28:00
