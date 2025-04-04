import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建文件处理器
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_model(model, save_path):
    """保存模型"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path):
    """加载模型"""
    # 检查文件是否存在
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型文件不存在: {load_path}")
    
    # 加载模型
    model.load_state_dict(torch.load(load_path))


def visualize_tsne(text_features, image_features, labels, save_path, title="特征t-SNE可视化"):
    """使用t-SNE可视化特征"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将特征转换为NumPy数组
    text_features = text_features.detach().cpu().numpy()
    image_features = image_features.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # 合并特征
    all_features = np.vstack([text_features, image_features])
    
    # 创建标签，区分文本和图像特征
    feature_types = np.array(['文本'] * len(text_features) + ['图像'] * len(image_features))
    
    # 创建情感标签
    sentiment_labels = np.concatenate([labels, labels])
    sentiment_map = {0: '负面', 1: '中性', 2: '正面'}
    sentiment_names = np.array([sentiment_map[label] for label in sentiment_labels])
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(all_features)
    
    # 创建DataFrame用于可视化
    df = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        '特征类型': feature_types,
        '情感': sentiment_names
    })
    
    # 绘制图像
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='x', y='y',
        hue='情感',
        style='特征类型',
        palette='viridis',
        data=df,
        alpha=0.7,
        s=100
    )
    
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_confusion_matrix(true_labels, pred_labels, save_path, class_names=None):
    """可视化混淆矩阵"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 设置类别名称
    if class_names is None:
        class_names = ['负面', '中性', '正面']
    
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建DataFrame
    df_cm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=cm, fmt='d', cmap='Blues', cbar=False)
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    """设置随机种子，确保实验可重复"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_label_distribution(labels):
    """获取标签分布"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    distribution = {label: count for label, count in zip(unique_labels, counts)}
    return distribution 