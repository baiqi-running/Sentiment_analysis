import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models
from ..config.config import Config

class CrossAttention(nn.Module):
    """交叉注意力模块，实现两个模态之间的交互"""
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        
    def forward(self, query, key, value):
        """
        Args:
            query: [batch_size, query_seq_len, query_dim]
            key: [batch_size, key_seq_len, key_dim]
            value: [batch_size, key_seq_len, value_dim]
        Returns:
            attended_features: [batch_size, query_seq_len, hidden_dim]
        """
        # 重塑输入张量以确保维度正确
        if len(query.shape) == 2:
            query = query.unsqueeze(1)
        if len(key.shape) == 2:
            key = key.unsqueeze(1)
        if len(value.shape) == 2:
            value = value.unsqueeze(1)
            
        Q = self.query_proj(query)  # [batch_size, query_seq_len, hidden_dim]
        K = self.key_proj(key)      # [batch_size, key_seq_len, hidden_dim]
        V = self.value_proj(value)  # [batch_size, key_seq_len, hidden_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权求和
        attended_features = torch.matmul(attention_weights, V)
        
        return attended_features

class DynamicGating(nn.Module):
    """动态门控机制，根据输入特征动态调整不同模态的权重"""
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(DynamicGating, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
            alpha: [batch_size, 1] - 文本模态的权重
        """
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # 计算门控值
        concat_features = torch.cat([text_proj, image_proj], dim=1)
        alpha = self.gate_net(concat_features)  # [batch_size, 1]
        
        # 加权融合
        fused_features = alpha * text_proj + (1 - alpha) * image_proj
        
        return fused_features, alpha

class AdaptiveFusionModel(nn.Module):
    """自适应融合模型，实现多模态情感分析"""
    def __init__(self, config):
        super(AdaptiveFusionModel, self).__init__()
        self.config = config
        
        # 文本编码器 (BERT)
        self.text_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # 图像编码器 (ResNet-50)
        vision_model = models.resnet50(pretrained=True)
        modules = list(vision_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_dim = config.IMAGE_EMBED_DIM
        
        # 特征投影层
        self.text_proj = nn.Linear(self.text_dim, config.FUSION_DIM)
        self.image_proj = nn.Linear(self.image_dim, config.FUSION_DIM)
        
        # 双路交叉注意力
        self.text_to_image_attention = CrossAttention(
            query_dim=config.FUSION_DIM,
            key_dim=config.FUSION_DIM,
            value_dim=config.FUSION_DIM,
            hidden_dim=config.FUSION_DIM
        )
        
        self.image_to_text_attention = CrossAttention(
            query_dim=config.FUSION_DIM,
            key_dim=config.FUSION_DIM,
            value_dim=config.FUSION_DIM,
            hidden_dim=config.FUSION_DIM
        )
        
        # 动态门控机制
        self.dynamic_gating = DynamicGating(
            text_dim=config.FUSION_DIM,
            image_dim=config.FUSION_DIM,
            hidden_dim=config.FUSION_DIM
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, config.NUM_CLASSES)
        )
        
        self.stochastic_depth = config.STOCHASTIC_DEPTH
        self.layer_drop = config.LAYER_DROP
        
    def _stochastic_depth(self, x, layer, p=0.1):
        """随机深度"""
        if self.training and p > 0:
            if torch.rand(1) < p:
                return x
        return layer(x)
    
    def _layer_drop(self, x, layer, p=0.1):
        """层dropout"""
        if self.training and p > 0:
            if torch.rand(1) < p:
                return x
        return layer(x)
    
    def encode_text(self, input_ids, attention_mask):
        """文本编码"""
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, text_dim]
        text_seq_features = outputs.last_hidden_state  # [batch_size, seq_len, text_dim]
        return text_features, text_seq_features
    
    def encode_image(self, images):
        """图像编码"""
        batch_size = images.size(0)
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)  # [batch_size, image_dim]
        return image_features, image_features.unsqueeze(1)  # 返回 [batch_size, image_dim] 和 [batch_size, 1, image_dim]
    
    def forward(self, input_ids, attention_mask, images):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            images: [batch_size, 3, 224, 224]
        Returns:
            logits: [batch_size, num_classes]
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            alpha: [batch_size, 1] - 文本模态的权重
        """
        # 编码文本和图像
        text_features, text_seq_features = self.encode_text(input_ids, attention_mask)
        image_features, image_seq_features = self.encode_image(images)
        
        # 应用随机深度
        text_features = self._stochastic_depth(text_features, lambda x: x, self.stochastic_depth)
        image_features = self._stochastic_depth(image_features, lambda x: x, self.stochastic_depth)
        
        # 首先进行特征投影到相同维度
        text_proj = self.text_proj(text_features)  # [batch_size, fusion_dim]
        image_proj = self.image_proj(image_features)  # [batch_size, fusion_dim]
        
        # 将序列特征投影到相同的维度
        text_seq_proj = self.text_proj(text_seq_features)  # [batch_size, seq_len, fusion_dim]
        # 处理图像序列特征
        batch_size = image_features.size(0)
        image_seq_proj = image_proj.view(batch_size, 1, -1)  # [batch_size, 1, fusion_dim]
        
        # 交叉注意力
        attended_image_features = self._layer_drop(
            text_seq_proj,
            lambda x: self.text_to_image_attention(
                query=x,
                key=image_seq_proj,
                value=image_seq_proj
            ),
            self.layer_drop
        )
        
        attended_text_features = self._layer_drop(
            image_seq_proj,
            lambda x: self.image_to_text_attention(
                query=x,
                key=text_seq_proj,
                value=text_seq_proj
            ),
            self.layer_drop
        )
        
        # 提取注意力后的特征
        attended_image_features = attended_image_features.mean(dim=1)  # [batch_size, fusion_dim]
        attended_text_features = attended_text_features.mean(dim=1)  # [batch_size, fusion_dim]
        
        # 特征增强（现在维度都是 fusion_dim）
        enhanced_text_features = text_proj + attended_text_features
        enhanced_image_features = image_proj + attended_image_features
        
        # 动态门控融合
        fused_features, alpha = self.dynamic_gating(enhanced_text_features, enhanced_image_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, text_features, image_features, alpha
    
    def get_contrastive_loss(self, text_features, image_features, temperature=0.07):
        """计算对比损失"""
        batch_size = text_features.size(0)
        
        # 将特征映射到相同的维度
        text_features = self.text_proj(text_features)  # [batch_size, fusion_dim]
        image_features = self.image_proj(image_features)  # [batch_size, fusion_dim]
        
        # 标准化特征
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.matmul(text_features, image_features.t()) / temperature
        
        # 对角线上的元素是正样本对
        labels = torch.arange(batch_size).to(logits.device)
        
        # 对比损失
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        loss = loss / 2.0
        
        return loss

class BaselineConcat(nn.Module):
    """基线模型：特征拼接"""
    def __init__(self, config):
        super(BaselineConcat, self).__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # 图像编码器
        vision_model = models.resnet50(pretrained=True)
        modules = list(vision_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_dim = config.IMAGE_EMBED_DIM
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim + self.image_dim, config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, config.NUM_CLASSES)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # 编码文本
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # 编码图像
        batch_size = images.size(0)
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)
        
        # 特征拼接
        concat_features = torch.cat([text_features, image_features], dim=1)
        
        # 分类
        logits = self.classifier(concat_features)
        
        return logits, text_features, image_features, None

class BaselineWeightedSum(nn.Module):
    """基线模型：加权平均"""
    def __init__(self, config):
        super(BaselineWeightedSum, self).__init__()
        self.config = config
        self.alpha = 0.5  # 固定权重
        
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # 图像编码器
        vision_model = models.resnet50(pretrained=True)
        modules = list(vision_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_dim = config.IMAGE_EMBED_DIM
        
        # 特征映射
        self.text_proj = nn.Linear(self.text_dim, config.FUSION_DIM)
        self.image_proj = nn.Linear(self.image_dim, config.FUSION_DIM)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, config.NUM_CLASSES)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # 编码文本
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_features)
        
        # 编码图像
        batch_size = images.size(0)
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)
        image_proj = self.image_proj(image_features)
        
        # 加权平均
        fused_features = self.alpha * text_proj + (1 - self.alpha) * image_proj
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, text_features, image_features, torch.tensor([self.alpha]).expand(batch_size, 1).to(images.device)

class ClipBasedFusion(nn.Module):
    """基线模型：基于CLIP的融合"""
    def __init__(self, config):
        super(ClipBasedFusion, self).__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # 图像编码器
        vision_model = models.resnet50(pretrained=True)
        modules = list(vision_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_dim = config.IMAGE_EMBED_DIM
        
        # 特征映射
        self.text_proj = nn.Linear(self.text_dim, config.FUSION_DIM)
        self.image_proj = nn.Linear(self.image_dim, config.FUSION_DIM)
        
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.FUSION_DIM * 2, config.FUSION_DIM),
            nn.ReLU(),
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, config.NUM_CLASSES)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # 编码文本
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_features)
        
        # 编码图像
        batch_size = images.size(0)
        image_features = self.image_encoder(images)
        image_features = image_features.view(batch_size, -1)
        image_proj = self.image_proj(image_features)
        
        # 归一化特征（类似CLIP）
        text_proj = F.normalize(text_proj, p=2, dim=1)
        image_proj = F.normalize(image_proj, p=2, dim=1)
        
        # 计算相似度
        similarity = torch.matmul(text_proj, image_proj.t())
        
        # MLP融合
        concat_features = torch.cat([text_proj, image_proj], dim=1)
        fused_features = self.fusion_mlp(concat_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, text_features, image_features, None 