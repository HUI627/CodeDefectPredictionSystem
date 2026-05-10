"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器类"""

    def __init__(self, config):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.processed_data_dir = config.processed_data_dir

    def load_and_prepare_data(self):
        """加载并准备数据"""
        logger.info("开始加载数据...")

        # 检查是否已有处理好的数据
        if self._check_processed_data_exists():
            logger.info("发现已处理的数据，直接加载...")
            train_df, val_df, test_df, stats = self.load_processed_data()
            logger.info(f"数据加载完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df

        # 尝试下载公开数据集
        logger.info("尝试下载公开数据集...")
        data = self._download_dataset()

        if data is None:
            # 如果下载失败，使用示例数据
            logger.warning("数据集下载失败，使用示例数据...")
            data = self._create_sample_data()

        # 预处理和划分数据
        train_data, val_data, test_data = self._split_data(data)

        # 计算统计信息
        stats = self._compute_statistics(train_data, val_data, test_data)

        # 保存处理后的数据
        self._save_processed_data(train_data, val_data, test_data, stats)

        logger.info(f"数据准备完成: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        return pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

    def _check_processed_data_exists(self):
        """检查处理后的数据是否存在"""
        return (self.processed_data_dir / 'train.csv').exists() and \
               (self.processed_data_dir / 'val.csv').exists() and \
               (self.processed_data_dir / 'test.csv').exists()

    def _download_dataset(self):
        """尝试下载公开数据集"""
        try:
            from datasets import load_dataset
            logger.info("正在下载数据集...")
            dataset = load_dataset("code_x_glue_cc_defect_detection", split='train')
            data = []
            for item in dataset:
                data.append({
                    'code': item['func'],
                    'label': item['target']
                })
            logger.info(f"✓ 数据集下载成功，共{len(data)}条数据")
            return data
        except Exception as e:
            logger.warning(f"数据集下载失败: {str(e)}")
            return None

    def _create_sample_data(self):
        """创建示例数据"""
        logger.info("创建示例数据集...")
        sample_codes = [
            ("def divide(a, b): return a / b", 1),
            ("def add(a, b): return a + b", 0),
            ("def get_item(lst, idx): return lst[idx]", 1),
            ("def multiply(x, y): return x * y", 0),
            ("def open_file(path): return open(path)", 1),
            ("def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b", 0),
            ("def process(data): return eval(data)", 1),
            ("def concat(a, b): return str(a) + str(b)", 0),
        ]

        data = []
        for code, label in sample_codes * 100:
            data.append({'code': code, 'label': label})

        logger.info(f"✓ 创建了{len(data)}条示例数据")
        return data

    def _split_data(self, data):
        """划分数据集"""
        train_ratio = self.config.train_split
        val_ratio = self.config.val_split
        test_ratio = self.config.test_split

        train_data, temp_data = train_test_split(
            data, test_size=(1 - train_ratio), random_state=self.config.random_seed
        )

        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size), random_state=self.config.random_seed
        )

        return train_data, val_data, test_data

    def _compute_statistics(self, train_data, val_data, test_data):
        """计算数据集统计信息"""
        stats = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_size': len(train_data) + len(val_data) + len(test_data)
        }
        return stats

    def _save_processed_data(self, train_data, val_data, test_data, stats):
        """保存处理后的数据"""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(train_data).to_csv(self.processed_data_dir / 'train.csv', index=False)
        pd.DataFrame(val_data).to_csv(self.processed_data_dir / 'val.csv', index=False)
        pd.DataFrame(test_data).to_csv(self.processed_data_dir / 'test.csv', index=False)

        with open(self.processed_data_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"✓ 处理后的数据已保存到 {self.processed_data_dir}")

    def load_processed_data(self):
        """加载处理后的数据"""
        train_df = pd.read_csv(self.processed_data_dir / 'train.csv')
        val_df = pd.read_csv(self.processed_data_dir / 'val.csv')
        test_df = pd.read_csv(self.processed_data_dir / 'test.csv')

        with open(self.processed_data_dir / 'stats.json', 'r') as f:
            stats = json.load(f)

        return train_df, val_df, test_df, stats
