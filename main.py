import os
import argparse
import logging
import torch
from src.config.config import Config
from src.data.dataset import get_data_loaders
from src.models.model import AdaptiveFusionModel, BaselineConcat, BaselineWeightedSum, ClipBasedFusion
from src.train.trainer import Trainer
from src.evaluation.ablation import AblationExperiment, create_ablation_models
from src.utils.utils import setup_logger, count_parameters, set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于自适应融合机制的图文多模态情感分析')
    parser.add_argument('--dataset', type=str, default='twitter2015', choices=['twitter2015', 'twitter2017'],
                        help='选择数据集: twitter2015 或 twitter2017')
    parser.add_argument('--model', type=str, default='adaptive_fusion',
                        choices=['adaptive_fusion', 'concat', 'weighted_sum', 'clip'],
                        help='选择模型: adaptive_fusion, concat, weighted_sum, 或 clip')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置全局日志记录器
    logger = setup_logger('main', os.path.join(config.LOG_DIR, 'main.log'))
    logger.info("开始运行图文多模态情感分析项目")
    logger.info(f"选择数据集: {args.dataset}")
    logger.info(f"选择模型: {args.model}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮次: {args.epochs}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"设备: {config.DEVICE}")
    
    # 加载数据
    logger.info("加载数据集...")
    data_loaders = get_data_loaders(config, dataset_name=args.dataset)
    train_loader, val_loader, test_loader, noise_test_loader = data_loaders
    
    # 创建模型
    logger.info("创建模型...")
    if args.model == 'adaptive_fusion':
        model = AdaptiveFusionModel(config)
        model_name = "自适应融合模型"
    elif args.model == 'concat':
        model = BaselineConcat(config)
        model_name = "特征拼接基线"
    elif args.model == 'weighted_sum':
        model = BaselineWeightedSum(config)
        model_name = "加权平均基线"
    elif args.model == 'clip':
        model = ClipBasedFusion(config)
        model_name = "CLIP融合基线"
    
    logger.info(f"使用模型: {model_name}")
    logger.info(f"模型参数总数: {count_parameters(model):,}")
    
    # 如果运行消融实验
    if args.ablation:
        logger.info("运行消融实验...")
        model_variants = create_ablation_models(config)
        ablation_experiment = AblationExperiment(config, data_loaders, model_variants)
        results = ablation_experiment.run()
        logger.info("消融实验完成")
        return
    
    # 训练模型
    logger.info("开始训练模型...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, noise_test_loader, config)
    training_results = trainer.train()
    
    # 输出结果
    logger.info("训练完成")
    logger.info("最终结果:")
    logger.info(f"测试集 - 准确率: {training_results['test_results']['accuracy']:.4f}")
    logger.info(f"测试集 - F1分数: {training_results['test_results']['f1']:.4f}")
    logger.info(f"测试集 - AUC: {training_results['test_results']['auc']:.4f}")
    logger.info(f"噪声测试集 - 准确率: {training_results['noise_test_results']['accuracy']:.4f}")
    logger.info(f"噪声测试集 - F1分数: {training_results['noise_test_results']['f1']:.4f}")
    logger.info(f"噪声测试集 - AUC: {training_results['noise_test_results']['auc']:.4f}")
    

if __name__ == "__main__":
    main() 