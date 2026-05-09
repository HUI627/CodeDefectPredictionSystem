"""
验证本地模型是否正确设置
运行此脚本检查模型文件是否完整且可以正常加载
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys

def verify_model():
    base_dir = Path(__file__).parent.absolute()
    model_path = base_dir / "models" / "codebert-base"

    print("=" * 60)
    print("验证本地 CodeBERT 模型设置")
    print("=" * 60)
    print(f"\n模型路径: {model_path}")

    # 检查目录是否存在
    if not model_path.exists():
        print(f"\n✗ 错误: 模型目录不存在")
        print(f"  请先运行 download_model.py 下载模型")
        return False

    print(f"✓ 模型目录存在")

    # 检查必需文件
    print("\n检查必需文件:")
    required_files = {
        "config.json": "模型配置文件",
        "pytorch_model.bin": "模型权重文件",
        "tokenizer_config.json": "分词器配置",
        "vocab.txt": "词汇表"
    }

    all_files_exist = True
    for file, description in required_files.items():
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {file:<25} ({description}, {size:.2f} MB)")
        else:
            print(f"  ✗ {file:<25} (缺失)")
            all_files_exist = False

    if not all_files_exist:
        print("\n✗ 部分文件缺失，请重新下载模型")
        return False

    # 尝试加载模型
    print("\n尝试加载模型:")
    try:
        print("  1/3 加载配置...")
        config = AutoConfig.from_pretrained(str(model_path))
        print(f"      ✓ 配置加载成功 (hidden_size={config.hidden_size})")

        print("  2/3 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print(f"      ✓ 分词器加载成功 (vocab_size={len(tokenizer)})")

        print("  3/3 加载模型...")
        model = AutoModel.from_pretrained(str(model_path))
        print(f"      ✓ 模型加载成功")

        # 测试推理
        print("\n测试模型推理:")
        test_code = "def hello(): print('world')"
        inputs = tokenizer(test_code, return_tensors="pt")
        outputs = model(**inputs)
        print(f"  ✓ 推理测试成功")
        print(f"    输入: {test_code}")
        print(f"    输出形状: {outputs.last_hidden_state.shape}")

        print("\n" + "=" * 60)
        print("✓ 所有检查通过！模型可以正常使用")
        print("=" * 60)
        print("\n你可以在代码中这样使用:")
        print(f'  model_name = "{model_path}"')
        print("  或")
        print(f'  model_name = str(Path(__file__).parent / "models" / "codebert-base")')

        return True

    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        print("\n可能的原因:")
        print("1. 模型文件损坏，请重新下载")
        print("2. transformers 版本不兼容，尝试: pip install --upgrade transformers")
        return False

if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)
