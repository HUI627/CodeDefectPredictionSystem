# 使用指南

## 快速开始（3步）

### 步骤1：测试环境

运行环境测试脚本，确保所有依赖都已正确安装：

```bash
python test_environment.py
```

如果测试失败，请先安装依赖：

```bash
pip install -r requirements.txt
```

### 步骤2：运行完整实验

**Windows用户**：
```bash
quick_start.bat
```

**Linux/Mac用户**：
```bash
bash quick_start.sh
```

或者直接运行：
```bash
python main.py --mode full
```

### 步骤3：查看结果

实验完成后，查看以下目录：
- `models/best_model.pt` - 训练好的模型
- `results/figures/` - 可视化图表
- `reports/` - 实验告报

## 详细使用说明

### 1. 仅训练模型

```bash
python main.py --mode train
```

这将：
- 下载并预处理数据
- 训练CodeBERT模型
- 保存最佳模型到 `models/best_model.pt`

### 2. 仅评估已训练的模型

```bash
python main.py --mode evaluate --model_path best_model.pt
```

这将：
- 加载训练好的模型
- 在测试集上评估
- 生成可视化图表和报告

### 3. 预测单个代码片段

```bash
python main.py --mode predict --code "def divide(a, b): return a / b"
```

输出示例：
```
Prediction: Defect
Confidence: 85.32%
Probabilities: No Defect=14.68%, Defect=85.32%
```

### 4. 预测代码文件

创建一个包含代码的文件 `test_code.txt`，然后：

```bash
python main.py --mode predict --code "$(cat test_code.txt)"
```

## 自定义配置

### 修改训练参数

编辑 `config.py` 文件：

```python
# 减小batch_size以节省内存
self.batch_size = 8

# 增加训练轮数
self.num_epochs = 10

# 调整学习率
self.learning_rate = 1e-5

# 使用CPU而不是GPU
self.device = "cpu"
```

### 使用自定义数据集

1. 准备CSV格式的数据文件，包含两列：
   - `code`: 代码字符串
   - `label`: 标签（0=无缺陷，1=有缺陷）

2. 将文件放到 `data/processed/` 目录：
   - `train.csv` - 训练集
   - `val.csv` - 验证集
   - `test.csv` - 测试集

3. 运行训练：
```bash
python main.py --mode train
```

## 常见使用场景

### 场景1：快速验证系统

```bash
# 1. 测试环境
python test_environment.py

# 2. 运行完整流程（使用示例数据）
python main.py --mode full

# 3. 查看报告
cd reports
# 打开最新的 .md 文件
```

### 场景2：使用真实数据集训练

```bash
# 1. 确保网络连接正常（需要下载数据集）
python main.py --mode train

# 2. 等待训练完成（可能需要1-2小时）

# 3. 评估模型
python main.py --mode evaluate

# 4. 查看结果
```

### 场景3：批量预测代码文件

创建一个Python脚本 `batch_predict.py`：

```python
from config import config
from src.model import CodeDefectPredictor
from transformers import AutoTokenizer
import torch

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = CodeDefectPredictor(config.model_name, config.num_labels, config.dropout_rate)
checkpoint = torch.load(config.model_save_dir / 'best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 读取代码文件
code_files = ['file1.py', 'file2.py', 'file3.py']

for file_path in code_files:
    with open(file_path, 'r') as f:
        code = f.read()

    # 预测
    encoding = tokenizer(code, max_length=512, truncation=True, 
                        padding='max_length', return_tensors='pt')
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        pred = torch.argmax(logits, dim=1).item()

    print(f"{file_path}: {'Defect' if pred == 1 else 'No Defect'}")
```

运行：
```bash
python batch_predict.py
```

## 性能优化技巧

### 1. GPU加速

确保使用GPU训练：
```python
# 在 config.py 中
self.device = "cuda"
```

检查GPU是否可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 调整batch_size

根据GPU内存调整：
- 8GB GPU: batch_size = 8-16
- 16GB GPU: batch_size = 16-32
- 24GB GPU: batch_size = 32-64

### 3. 使用梯度累积

如果GPU内存不足，使用梯度累积模拟更大的batch_size：

```python
# 在 config.py 中
self.batch_size = 8
self.gradient_accumulation_steps = 4  # 等效于 batch_size=32
```

### 4. 减少序列长度

如果代码片段通常较短：
```python
# 在 config.py 中
self.max_seq_length = 256  # 从512减少到256
```

## 故障排除

### 问题1：CUDA out of memory

**解决方案**：
1. 减小 `batch_size`
2. 减小 `max_seq_length`
3. 使用CPU训练（较慢）

### 问题2：下载数据集失败

**解决方案**：
1. 检查网络连接
2. 使用代理或VPN
3. 系统会自动使用示例数据集

### 问题3：训练速度很慢

**解决方案**：
1. 确认使用GPU（`device="cuda"`）
2. 增大 `batch_size`
3. 减少 `num_epochs`

### 问题4：模型过拟合

**解决方案**：
1. 增大 `dropout_rate`
2. 减少 `num_epochs`
3. 使用更多训练数据
4. 启用数据增强

## 进阶使用

### 1. 模型集成

训练多个模型并集成预测结果：

```python
# 训练3个不同随机种子的模型
for seed in [42, 123, 456]:
    config.random_seed = seed
    trainer = Trainer(config)
    trainer.train(train_df, val_df)
    trainer.save_model(f'model_seed_{seed}.pt')

# 集成预测
predictions = []
for seed in [42, 123, 456]:
    model.load_state_dict(torch.load(f'models/model_seed_{seed}.pt'))
    pred = model.predict(test_data)
    predictions.append(pred)

# 投票
final_pred = np.mean(predictions, axis=0) > 0.5
```

### 2. 超参数调优

使用网格搜索找到最佳参数：

```python
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [8, 16, 32]

best_f1 = 0
best_params = {}

for lr in learning_rates:
    for bs in batch_sizes:
        config.learning_rate = lr
        config.batch_size = bs

        trainer = Trainer(config)
        trainer.train(train_df, val_df)

        evaluator = Evaluator(config, trainer.model, tokenizer)
        metrics, _, _, _ = evaluator.evaluate(val_df)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_params = {'lr': lr, 'bs': bs}

print(f"Best params: {best_params}, F1: {best_f1}")
```

### 3. 注意力可视化

分析模型关注的代码部分：

```python
# 获取注意力权重
outputs = model.codebert(input_ids, attention_mask, output_attentions=True)
attentions = outputs.attentions  # 所有层的注意力权重

# 可视化最后一层的注意力
import matplotlib.pyplot as plt
import seaborn as sns

last_layer_attention = attentions[-1][0].mean(dim=0).detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

plt.figure(figsize=(10, 10))
sns.heatmap(last_layer_attention, xticklabels=tokens, yticklabels=tokens)
plt.title('Attention Weights')
plt.savefig('attention_visualization.png')
```

## 获取帮助

如果遇到问题：
1. 查看本文档的"故障排除"部分
2. 查看 `README.md` 中的常见问题
3. 运行 `python test_environment.py` 检查环境
4. 查看日志文件了解详细错误信息

---

**最后更新**: 2026年5月
