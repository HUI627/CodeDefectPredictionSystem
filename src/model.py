import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class CodeDefectPredictor(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
        super(CodeDefectPredictor, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
