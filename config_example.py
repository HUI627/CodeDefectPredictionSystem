# 配置示例文件
# 复制此文件为 config_custom.py 并根据需要修改

import os
from pathlib import Path

class CustomConfig:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()

        # ========== 模型配置 ==========
        # 可选模型：
        # - microsoft/codebert-base (推荐)
        # - microsoft/graphcodebert-base (更强大但更慢)
        # - huggingface/CodeBERTa-small-v1 (更小更快)
        self.model_name = "microsoft/codebert-base"
        self.num_labels = 2  # 二分类
        self.hidden_size = 768
        self.dropout_rate = 0.1  # 增大可以减少过拟合

        # ========== 训练配置 ==========
        self.max_seq_length = 512  # 减小可以节省内存
        self.batch_size = 16  # GPU内存不足时减小此值
        self.learning_rate = 2e-5  # 典型范围：1e-5 到 5e-5
        self.num_epochs = 5  # 增加可能提升性能但容易过拟合
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.gradient_accumulation_steps = 1  # 增大可以模拟更大的batch_size

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
        self.early_stopping_patience = 3  # 验证集loss不下降的容忍轮数
        self.save_best_model = True
        self.random_seed = 42

        # ========== 硬件配置 ==========
        # 选项：'cuda' (GPU) 或 'cpu'
        self.device = "cuda"  # 如果没有GPU，改为 "cpu"
        self.num_workers = 4  # 数据加载的进程数

        # ========== 日志配置 ==========
        self.log_interval = 10  # 每N个batch打印一次日志
        self.eval_interval = 100  # 每N个batch评估一次

# 使用示例：
# from config_custom import CustomConfig
# config = CustomConfig()
