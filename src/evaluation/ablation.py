import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ..config.config import Config
from ..models.model import AdaptiveFusionModel, BaselineConcat, BaselineWeightedSum, ClipBasedFusion
from ..utils.utils import setup_logger, save_model, load_model, visualize_tsne, visualize_confusion_matrix


class AblationExperiment:
    """消融实验类，用于比较不同模型配置的性能"""
    
    def __init__(self, config, data_loaders, model_variants):
        """
        初始化消融实验
        
        Args:
            config: 配置对象
            data_loaders: 包含训练、验证、测试和噪声测试数据加载器的元组
            model_variants: 要比较的模型变体列表
        """
        self.config = config
        self.train_loader, self.val_loader, self.test_loader, self.noise_test_loader = data_loaders
        self.model_variants = model_variants
        self.device = config.DEVICE
        
        # 设置日志记录器
        self.logger = setup_logger('ablation', os.path.join(config.LOG_DIR, 'ablation.log'))
        
        # 保存结果的字典
        self.results = {}
        
    def run(self):
        """运行消融实验"""
        self.logger.info("开始消融实验")
        
        # 创建结果目录
        result_dir = os.path.join(self.config.RESULT_DIR, 'ablation')
        os.makedirs(result_dir, exist_ok=True)
        
        # 按模型变体运行实验
        for variant_name, model in self.model_variants.items():
            self.logger.info(f"评估模型变体: {variant_name}")
            
            # 将模型移到设备上
            model.to(self.device)
            
            # 在测试集上评估
            test_results = self.evaluate(model, self.test_loader, f"{variant_name} - 测试集")
            
            # 在噪声测试集上评估
            noise_test_results = self.evaluate(model, self.noise_test_loader, f"{variant_name} - 噪声测试集")
            
            # 保存结果
            self.results[variant_name] = {
                'test': test_results,
                'noise_test': noise_test_results
            }
            
            # 保存特征可视化
            if hasattr(model, 'encode_text') and hasattr(model, 'encode_image'):
                self.visualize_features(model, variant_name, result_dir)
            
            # 保存混淆矩阵
            self.save_confusion_matrix(test_results, variant_name, result_dir, suffix='test')
            self.save_confusion_matrix(noise_test_results, variant_name, result_dir, suffix='noise_test')
        
        # 比较和汇总结果
        self.compare_results(result_dir)
        
        return self.results
    
    def evaluate(self, model, data_loader, data_name):
        """评估模型性能"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_text_features = []
        all_image_features = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"评估{data_name}"):
                # 将数据移至设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = model(input_ids, attention_mask, images)
                
                # 获取预测结果
                logits = outputs[0]  # 假设第一个输出是logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                # 收集预测和标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 收集特征（如果有）
                if len(outputs) > 1:
                    text_features = outputs[1]  # 假设第二个输出是文本特征
                    image_features = outputs[2]  # 假设第三个输出是图像特征
                    all_text_features.append(text_features)
                    all_image_features.append(image_features)
        
        # 转换为NumPy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 计算AUC-ROC（多分类）
        auc = 0
        try:
            auc = roc_auc_score(
                np.eye(self.config.NUM_CLASSES)[all_labels],  # 转换为one-hot编码
                all_probs,
                multi_class='ovr',
                average='macro'
            )
        except Exception as e:
            self.logger.warning(f"计算AUC时出错: {e}")
        
        # 记录结果
        self.logger.info(f"{data_name} 评估结果:")
        self.logger.info(f"准确率: {accuracy:.4f}")
        self.logger.info(f"F1分数(宏平均): {f1:.4f}")
        self.logger.info(f"AUC-ROC: {auc:.4f}")
        
        # 如果有收集特征，则合并它们
        text_features = None
        image_features = None
        if all_text_features and all_image_features:
            text_features = torch.cat(all_text_features, dim=0)
            image_features = torch.cat(all_image_features, dim=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'text_features': text_features,
            'image_features': image_features
        }
    
    def visualize_features(self, model, variant_name, result_dir):
        """使用t-SNE可视化特征"""
        self.logger.info(f"为模型变体生成特征可视化: {variant_name}")
        
        # 收集样本特征
        model.eval()
        sample_text_features = []
        sample_image_features = []
        sample_labels = []
        
        with torch.no_grad():
            # 从测试集中获取一些样本
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 获取特征
                _, text_features, image_features, _ = model(input_ids, attention_mask, images)
                
                sample_text_features.append(text_features)
                sample_image_features.append(image_features)
                sample_labels.append(labels)
                
                # 限制样本数量
                if len(sample_labels) >= 5:  # 收集5个批次的样本
                    break
        
        # 合并特征
        if sample_text_features and sample_image_features:
            text_features = torch.cat(sample_text_features, dim=0)
            image_features = torch.cat(sample_image_features, dim=0)
            labels = torch.cat(sample_labels, dim=0)
            
            # 生成可视化
            save_path = os.path.join(result_dir, f"{variant_name}_tsne.png")
            visualize_tsne(text_features, image_features, labels, save_path, 
                          title=f"{variant_name} - 特征分布 (t-SNE)")
    
    def save_confusion_matrix(self, results, variant_name, result_dir, suffix='test'):
        """保存混淆矩阵"""
        if 'predictions' in results and 'labels' in results:
            save_path = os.path.join(result_dir, f"{variant_name}_confusion_matrix_{suffix}.png")
            visualize_confusion_matrix(results['labels'], results['predictions'], save_path)
    
    def compare_results(self, result_dir):
        """比较和汇总不同模型变体的结果"""
        self.logger.info("比较模型变体性能")
        
        # 创建用于比较的DataFrame
        test_metrics = []
        noise_test_metrics = []
        
        for variant_name, results in self.results.items():
            # 测试集结果
            test_metrics.append({
                '模型': variant_name,
                '数据集': '测试集',
                '准确率': results['test']['accuracy'],
                'F1分数': results['test']['f1'],
                'AUC': results['test']['auc']
            })
            
            # 噪声测试集结果
            noise_test_metrics.append({
                '模型': variant_name,
                '数据集': '噪声测试集',
                '准确率': results['noise_test']['accuracy'],
                'F1分数': results['noise_test']['f1'],
                'AUC': results['noise_test']['auc']
            })
        
        # 合并结果
        all_metrics = pd.DataFrame(test_metrics + noise_test_metrics)
        
        # 保存为CSV
        csv_path = os.path.join(result_dir, 'model_comparison.csv')
        all_metrics.to_csv(csv_path, index=False)
        
        # 绘制比较图表
        self._plot_metric_comparison(all_metrics, '准确率', result_dir)
        self._plot_metric_comparison(all_metrics, 'F1分数', result_dir)
        self._plot_metric_comparison(all_metrics, 'AUC', result_dir)
        
        # 输出比较结果
        self.logger.info(f"结果比较已保存到 {csv_path}")
    
    def _plot_metric_comparison(self, df, metric, result_dir):
        """绘制指标比较图"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='模型', y=metric, hue='数据集', data=df)
        plt.title(f"{metric}比较")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{metric}_comparison.png"))
        plt.close()


def create_ablation_models(config):
    """创建用于消融实验的模型变体"""
    # 完整模型
    full_model = AdaptiveFusionModel(config)
    
    # 基线模型
    concat_model = BaselineConcat(config)
    weighted_sum_model = BaselineWeightedSum(config)
    clip_model = ClipBasedFusion(config)
    
    # 消融变体：移除双路交叉注意力（仅保留单路注意力）
    class SinglePathAttentionModel(AdaptiveFusionModel):
        def forward(self, input_ids, attention_mask, images):
            # 编码文本和图像
            text_features, text_seq_features = self.encode_text(input_ids, attention_mask)
            image_features, image_seq_features = self.encode_image(images)
            
            # 只使用文本->图像的注意力
            attended_image_features = self.text_to_image_attention(
                query=text_seq_features,
                key=image_seq_features,
                value=image_seq_features
            )
            
            # 提取注意力后的特征
            attended_image_features = attended_image_features[:, 0, :]
            
            # 投影到共同的特征空间
            text_proj = self.text_proj(text_features)
            image_proj = self.image_proj(image_features)
            
            # 特征增强（只增强图像特征）
            enhanced_text_features = text_proj
            enhanced_image_features = image_proj + attended_image_features
            
            # 动态门控融合
            fused_features, alpha = self.dynamic_gating(enhanced_text_features, enhanced_image_features)
            
            # 分类
            logits = self.classifier(fused_features)
            
            return logits, text_features, image_features, alpha
    
    single_path_model = SinglePathAttentionModel(config)
    
    # 消融变体：移除动态门控（固定权重）
    class FixedGatingModel(AdaptiveFusionModel):
        def forward(self, input_ids, attention_mask, images):
            # 编码文本和图像
            text_features, text_seq_features = self.encode_text(input_ids, attention_mask)
            image_features, image_seq_features = self.encode_image(images)
            
            # 交叉注意力
            attended_image_features = self.text_to_image_attention(
                query=text_seq_features,
                key=image_seq_features,
                value=image_seq_features
            )
            
            attended_text_features = self.image_to_text_attention(
                query=image_seq_features,
                key=text_seq_features,
                value=text_seq_features
            )
            
            # 提取注意力后的特征
            attended_image_features = attended_image_features[:, 0, :]
            attended_text_features = attended_text_features[:, 0, :]
            
            # 投影到共同的特征空间
            text_proj = self.text_proj(text_features)
            image_proj = self.image_proj(image_features)
            
            # 特征增强
            enhanced_text_features = text_proj + attended_text_features
            enhanced_image_features = image_proj + attended_image_features
            
            # 固定权重融合 (α=0.5)
            fixed_alpha = 0.5
            fused_features = fixed_alpha * enhanced_text_features + (1 - fixed_alpha) * enhanced_image_features
            alpha = torch.tensor([fixed_alpha]).expand(images.size(0), 1).to(images.device)
            
            # 分类
            logits = self.classifier(fused_features)
            
            return logits, text_features, image_features, alpha
    
    fixed_gating_model = FixedGatingModel(config)
    
    # 消融变体：移除对比损失
    # 注意：这只需要在训练时修改损失计算，模型架构不变
    # 所以我们仍使用完整模型，但在训练时不应用对比损失
    
    return {
        '完整模型': full_model,
        '特征拼接 (Concat)': concat_model,
        '加权平均 (Weighted Sum)': weighted_sum_model,
        'CLIP融合': clip_model,
        '单路注意力': single_path_model,
        '固定门控': fixed_gating_model
    } 