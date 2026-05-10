#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码缺陷预测系统 - 主程序
支持训练、评估、预测三种模式
"""

import argparse
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from src.utils import setup_logging, set_seed, ensure_dir
from src.data_loader import DataLoader
from src.dataset import CodeDefectDataset
from src.model import CodeDefectPredictor
from src.train import Trainer
from src.evaluate import Evaluator
from src.visualize import Visualizer
from src.report_generator import ReportGenerator
from transformers import AutoTokenizer

logger = setup_logging()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='代码缺陷预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程（训练+评估+可视化+报告）
  python main.py --mode full

  # 仅训练模型
  python main.py --mode train

  # 仅评估模型
  python main.py --mode evaluate --model_path best_model.pt

  # 预测单个代码片段
  python main.py --mode predict --code "def divide(a, b): return a / b"
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['train', 'evaluate', 'predict', 'full'],
        help='运行模式: train(训练), evaluate(评估), predict(预测), full(完整流程)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='best_model.pt',
        help='模型文件名（用于evaluate和predict模式）'
    )

    parser.add_argument(
        '--code',
        type=str,
        default=None,
        help='要预测的代码片段（用于predict模式）'
    )

    return parser.parse_args()

def train_model(train_df, val_df, tokenizer):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("开始训练模型...")
    logger.info("=" * 60)

    # 创建数据集
    train_dataset = CodeDefectDataset(train_df, tokenizer, config.max_seq_length)
    val_dataset = CodeDefectDataset(val_df, tokenizer, config.max_seq_length)

    # 创建模型
    model = CodeDefectPredictor(config.model_name, config.num_labels, config.dropout_rate)

    # 创建训练器
    trainer = Trainer(config, model, train_dataset, val_dataset)

    # 训练模型
    trainer.train()

    # 保存训练历史
    trainer.save_training_history()

    logger.info("✓ 模型训练完成")
    return trainer

def evaluate_model(test_df, tokenizer, model_path='best_model.pt'):
    """评估模型"""
    logger.info("=" * 60)
    logger.info("开始评估模型...")
    logger.info("=" * 60)

    # 加载模型
    model = CodeDefectPredictor(config.model_name, config.num_labels, config.dropout_rate)
    checkpoint = torch.load(config.model_save_dir / model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ 模型已加载: {model_path}")

    # 创建测试数据集
    test_dataset = CodeDefectDataset(test_df, tokenizer, config.max_seq_length)

    # 创建评估器
    evaluator = Evaluator(config, model, tokenizer)

    # 评估模型
    metrics, predictions, labels, probs = evaluator.evaluate(test_df)

    # 保存评估结果
    evaluator.save_results(metrics, predictions, labels, probs)

    logger.info("✓ 模型评估完成")
    return metrics, predictions, labels, probs

def visualize_results(metrics, predictions, labels, probs):
    """生成可视化图表"""
    logger.info("=" * 60)
    logger.info("生成可视化图表...")
    logger.info("=" * 60)

    visualizer = Visualizer(config)

    # 生成所有图表
    visualizer.plot_training_history()
    visualizer.plot_confusion_matrix(labels, predictions)
    visualizer.plot_roc_curve(labels, probs)
    visualizer.plot_precision_recall_curve(labels, probs)
    visualizer.plot_class_distribution(labels, predictions)

    logger.info("✓ 可视化图表生成完成")

def generate_report(metrics):
    """生成实验报告"""
    logger.info("=" * 60)
    logger.info("生成实验报告...")
    logger.info("=" * 60)

    report_gen = ReportGenerator(config)
    report_path = report_gen.generate_report(metrics)

    logger.info(f"✓ 实验报告已生成: {report_path}")
    return report_path

def predict_code(code, tokenizer, model_path='best_model.pt'):
    """预测单个代码片段"""
    logger.info("=" * 60)
    logger.info("预测代码缺陷...")
    logger.info("=" * 60)

    # 加载模型
    model = CodeDefectPredictor(config.model_name, config.num_labels, config.dropout_rate)
    checkpoint = torch.load(config.model_save_dir / model_path,
                          map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Tokenize代码
    encoding = tokenizer(
        code,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 预测
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()

    # 输出结果
    logger.info(f"\n代码: {code}")
    logger.info(f"预测结果: {'有缺陷 (Defect)' if pred == 1 else '无缺陷 (No Defect)'}")
    logger.info(f"置信度: {probs[0][pred].item():.2%}")
    logger.info(f"概率分布: 无缺陷={probs[0][0].item():.2%}, 有缺陷={probs[0][1].item():.2%}")

def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(config.random_seed)

    # 创建必要的目录
    for directory in [config.model_save_dir, config.results_dir, config.figures_dir,
                      config.metrics_dir, config.reports_dir, config.data_dir,
                      config.raw_data_dir, config.processed_data_dir]:
        ensure_dir(directory)

    logger.info("=" * 60)
    logger.info("代码缺陷预测系统")
    logger.info("=" * 60)
    logger.info(f"运行模式: {args.mode}")

    try:
        # 加载tokenizer
        logger.info(f"加载tokenizer: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        logger.info("✓ Tokenizer加载成功")

        if args.mode == 'predict':
            # 预测模式
            if args.code is None:
                logger.error("错误: predict模式需要提供--code参数")
                sys.exit(1)
            predict_code(args.code, tokenizer, args.model_path)

        else:
            # 加载数据
            data_loader = DataLoader(config)
            train_df, val_df, test_df = data_loader.load_and_prepare_data()

            if args.mode == 'train':
                # 仅训练
                train_model(train_df, val_df, tokenizer)

            elif args.mode == 'evaluate':
                # 仅评估
                metrics, predictions, labels, probs = evaluate_model(
                    test_df, tokenizer, args.model_path
                )
                visualize_results(metrics, predictions, labels, probs)
                generate_report(metrics)

            elif args.mode == 'full':
                # 完整流程
                logger.info("\n执行完整流程: 训练 -> 评估 -> 可视化 -> 报告")

                # 1. 训练
                trainer = train_model(train_df, val_df, tokenizer)

                # 2. 评估
                metrics, predictions, labels, probs = evaluate_model(
                    test_df, tokenizer, 'best_model.pt'
                )

                # 3. 可视化
                visualize_results(metrics, predictions, labels, probs)

                # 4. 生成报告
                report_path = generate_report(metrics)

                logger.info("\n" + "=" * 60)
                logger.info("✓ 完整流程执行完成！")
                logger.info("=" * 60)
                logger.info(f"模型保存位置: {config.model_save_dir / 'best_model.pt'}")
                logger.info(f"可视化图表: {config.figures_dir}")
                logger.info(f"实验报告: {report_path}")
                logger.info("=" * 60)

        logger.info("\n程序执行成功！")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
