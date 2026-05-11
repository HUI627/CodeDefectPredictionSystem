"""
工具函数模块
"""
import random
import numpy as np
import torch
import logging
from pathlib import Path

def set_seed(seed):
    """
    设置所有随机种子以确保可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_file=None):
    """
    配置日志系统

    Args:
        log_file: 日志文件路径（可选）

    Returns:
        logger: 配置好的logger对象
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # 强制重新配置，避免重复配置问题
    )
    return logging.getLogger(__name__)

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径（str或Path对象）
    """
    if directory is None:
        raise ValueError("directory参数不能为None")
    Path(directory).mkdir(parents=True, exist_ok=True)

def count_parameters(model):
    """
    统计模型的可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        int: 可训练参数总数
    """
    if model is None:
        raise ValueError("model参数不能为None")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    """
    将秒数格式化为HH:MM:SS格式

    Args:
        seconds: 秒数（int或float）

    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 0:
        raise ValueError("seconds不能为负数")

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_device():
    """
    获取可用的计算设备

    Returns:
        torch.device: CUDA设备（如果可用）或CPU设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    return device

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    logging.info(f"检查点已保存到 {filepath}")
