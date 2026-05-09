# 项目配置文件
# 此文件包含所有训练和模型配置

import os
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()

        # ========== 模型配置 ==========
        # 使用本地模型路径（将下载的模型放在 models/codebert-base 文件夹中）
        self.model_name = str(self.base_dir / "models" / "codebert-base")

        # 如果要使用在线模型，取消注释下面这行并注释掉上面的本地路径：
        # self.model_name = "microsoft/codebert-base"

        self.num_labels = 2  # 二分类：有缺陷/无缺陷
        self.hidden_size = 768
        self.dropout_rate = 0.1

        # ========== 训练配置 ==========
        self.max_seq_length = 512
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 5
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 1

        # ========== 数据配置 ==========
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15

        # ========== 路径配置 ==========
        self.model_save_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.metrics_dir = self.results_dir / "metrics"
        self.reports_dir = self.base_dir / "reports"

        # ========== 训练策略 ==========
        self.early_stopping_patience = 3
        self.save_best_model = True
        self.random_seed = 42

        # ========== 硬件配置 ==========
        self.device = "cuda"
        self.num_workers = 4

        # ========== 日志配置 ==========
        self.log_interval = 10
        self.eval_interval = 100

config = Config()
