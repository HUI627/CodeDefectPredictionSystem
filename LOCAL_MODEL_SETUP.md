# 本地模型设置指南

## 1. 下载模型到本地

将 CodeBERT 模型下载到以下目录结构：

```
D:\研\人工智能\大作业\
└── models/
    └── codebert-base/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        ├── vocab.txt
        └── 其他模型文件...
```

## 2. 模型文件说明

确保 `models/codebert-base/` 文件夹中包含以下文件：
- `config.json` - 模型配置文件
- `pytorch_model.bin` - 模型权重文件
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表
- `tokenizer.json` (可选)
- `special_tokens_map.json` (可选)

## 3. 修改配置文件

### 方法一：使用 config_example.py（推荐）

`config_example.py` 已经修改为默认使用本地模型路径：

```python
self.model_name = str(self.base_dir / "models" / "codebert-base")
```

如果你想切换回在线下载模式，只需取消注释这行：
```python
# self.model_name = "microsoft/codebert-base"
```

### 方法二：直接修改 config.py

如果你的项目使用 `config.py`，添加以下内容：

```python
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        
        # 使用本地模型
        self.model_name = str(self.base_dir / "models" / "codebert-base")
        
        # 其他配置...
```

## 4. 验证设置

运行以下 Python 代码验证模型是否能正确加载：

```python
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# 设置本地模型路径
model_path = Path(__file__).parent / "models" / "codebert-base"

# 尝试加载
try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    print("✓ 模型加载成功！")
    print(f"模型路径: {model_path}")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
```

## 5. 常见问题

### Q: 如何下载 CodeBERT 模型？

**方法 1: 使用 Hugging Face CLI**
```bash
pip install huggingface_hub
huggingface-cli download microsoft/codebert-base --local-dir models/codebert-base
```

**方法 2: 使用 Python 脚本**
```python
from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/codebert-base"
save_path = "./models/codebert-base"

# 下载并保存
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```

### Q: 模型文件夹名称必须是 codebert-base 吗？

不是必须的。你可以使用任何名称，只需在配置文件中相应修改：
```python
self.model_name = str(self.base_dir / "models" / "你的文件夹名称")
```

### Q: 为什么要使用 str() 包装路径？

`transformers` 库的某些版本需要字符串路径而不是 Path 对象。使用 `str()` 确保兼容性。

## 6. 优势

使用本地模型的优势：
- ✓ 无需网络连接
- ✓ 加载速度更快
- ✓ 避免下载超时问题
- ✓ 便于版本控制和复现
