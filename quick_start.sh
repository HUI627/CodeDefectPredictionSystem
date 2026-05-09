#!/bin/bash

echo "=================================="
echo "代码缺陷预测系统 - 快速开始"
echo "=================================="

echo ""
echo "[1/3] 检查Python环境..."
python --version

echo ""
echo "[2/3] 安装依赖包..."
pip install -r requirements.txt

echo ""
echo "[3/3] 运行完整实验流程..."
python main.py --mode full

echo ""
echo "=================================="
echo "实验完成！"
echo "=================================="
echo ""
echo "查看结果："
echo "- 模型: models/best_model.pt"
echo "- 图表: results/figures/"
echo "- 报告: reports/"
echo ""
