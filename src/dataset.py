import torch
from torch.utils.data import Dataset

class CodeDefectDataset(Dataset):
    """代码缺陷数据集类"""

    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        初始化数据集

        Args:
            dataframe: pandas DataFrame，包含'code'和'label'列
            tokenizer: 预训练的tokenizer
            max_length: 最大序列长度
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 验证数据
        if 'code' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("DataFrame必须包含'code'和'label'列")

        # 移除空值
        self.data = self.data.dropna(subset=['code', 'label']).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            code = str(self.data.iloc[idx]['code'])
            label = int(self.data.iloc[idx]['label'])
        except (KeyError, ValueError, IndexError) as e:
            raise IndexError(f"无法访问索引{idx}的数据: {str(e)}")

        # Tokenize代码
        encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
