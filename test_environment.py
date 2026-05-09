"""
测试脚本 - 验证环境配置是否正确
"""

import sys

def test_imports():
    print("测试导入必要的库...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        import transformers
        print(f"✓ Transformers {transformers.__version__}")

        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")

        import pandas
        print(f"✓ Pandas {pandas.__version__}")

        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")

        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")

        import numpy
        print(f"✓ NumPy {numpy.__version__}")

        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_cuda():
    print("\n测试CUDA可用性...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA不可用，将使用CPU训练（速度较慢）")
        return True
    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        return False

def test_model_download():
    print("\n测试模型加载...")
    try:
        from pathlib import Path
        from transformers import AutoTokenizer

        # 尝试从本地加载模型
        base_dir = Path(__file__).parent.absolute()
        local_model_path = base_dir / "models" / "codebert-base"

        if local_model_path.exists():
            print(f"从本地加载模型: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
            print("✓ 本地模型加载成功")
        else:
            print("本地模型不存在，从在线下载（首次运行需要下载）...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            print("✓ 在线模型下载成功")
            print(f"  提示: 运行 'python download_model.py' 可下载到本地")

        test_code = "def add(a, b): return a + b"
        tokens = tokenizer(test_code, return_tensors="pt")
        print(f"✓ Tokenization测试成功")
        print(f"  输入代码: {test_code}")
        print(f"  Token数量: {tokens['input_ids'].shape[1]}")
        return True
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        print("  提示: 可能需要网络连接或代理")
        return False

def main():
    print("=" * 60)
    print("代码缺陷预测系统 - 环境测试")
    print("=" * 60)

    results = []
    results.append(("库导入", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("模型下载", test_model_download()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ 所有测试通过！可以开始使用系统。")
        print("\n运行以下命令开始训练：")
        print("  python main.py --mode full")
    else:
        print("\n✗ 部分测试失败，请检查环境配置。")
        print("\n安装依赖：")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
