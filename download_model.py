"""
下载 CodeBERT 模型到本地
运行此脚本将自动下载模型到 models/codebert-base 文件夹
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys

def download_model():
    # 设置路径
    base_dir = Path(__file__).parent.absolute()
    model_name = "microsoft/codebert-base"
    save_path = base_dir / "models" / "codebert-base"

    print(f"开始下载模型: {model_name}")
    print(f"保存路径: {save_path}")
    print("-" * 60)

    # 创建目录
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # 下载配置
        print("1/3 下载模型配置...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(save_path)
        print("✓ 配置下载完成")

        # 下载分词器
        print("\n2/3 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✓ 分词器下载完成")

        # 下载模型
        print("\n3/3 下载模型权重（这可能需要几分钟）...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print("✓ 模型权重下载完成")

        print("\n" + "=" * 60)
        print("✓ 所有文件下载成功！")
        print(f"模型已保存到: {save_path}")
        print("=" * 60)

        # 验证
        print("\n验证模型文件...")
        required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
        for file in required_files:
            file_path = save_path / file
            if file_path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (缺失)")

        return True

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的解决方案：")
        print("1. 检查网络连接")
        print("2. 确保已安装 transformers: pip install transformers")
        print("3. 如果在国内，可能需要配置代理或使用镜像源")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
