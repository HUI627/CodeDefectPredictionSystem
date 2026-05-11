import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class CodeDefectPredictor(nn.Module):
    """基于CodeBERT的代码缺陷预测模型"""

    def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
        """
        初始化模型

        Args:
            model_name: 预训练模型名称或路径
            num_labels: 分类标签数量（默认2：有缺陷/无缺陷）
            dropout_rate: Dropout比率
        """
        super(CodeDefectPredictor, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 初始化分类器权重
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        Args:
            input_ids: 输入token IDs [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]

        Returns:
            logits: 分类logits [batch_size, num_labels]
        """
        # 输入验证
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids应该是2维张量，但得到{input_ids.dim()}维")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask应该是2维张量，但得到{attention_mask.dim()}维")

        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用pooler_output，如果不存在则使用last_hidden_state的[CLS] token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # 使用[CLS] token的输出（第一个token）
            pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
