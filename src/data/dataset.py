import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer
import random
from ..config.config import Config

class TwitterDataset(Dataset):
    def __init__(self, config, dataset_name='twitter2015', split='train', add_noise=False):
        """
        初始化Twitter情感分析数据集
        Args:
            config: 配置对象
            dataset_name: 'twitter2015' 或 'twitter2017'
            split: 'train', 'val', 或 'test'
            add_noise: 是否添加噪声标签
        """
        self.config = config
        self.max_length = config.MAX_TEXT_LENGTH
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        self.split = split
        
        # 选择数据集路径
        if dataset_name == 'twitter2015':
            self.data_path = config.TWITTER15_PATH
            self.image_path = config.TWITTER15_IMAGES_PATH
        else:
            self.data_path = config.TWITTER17_PATH
            self.image_path = config.TWITTER17_IMAGES_PATH
        
        # 读取数据
        self.data = pd.read_csv(
            os.path.join(self.data_path, f"{split}.tsv"), 
            sep='\t', 
            skiprows=1,  # 跳过第一行
            header=None,
            names=['index', 'sentiment', 'image_id', 'text', 'entity']
        )
        
        # 标签映射 (0:负面, 1:中性, 2:正面)
        self.data['sentiment'] = self.data['sentiment'].astype(int)
        
        # 设置图像增强
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 为冲突样本添加噪声标签
        self.add_noise = add_noise
        if add_noise and split == 'test':
            # 在测试集中随机选择样本添加噪声，模拟文本和图像情感不一致的情况
            noise_indices = random.sample(range(len(self.data)), min(config.NOISE_TEST_SAMPLES, len(self.data)))
            for idx in noise_indices:
                # 翻转标签 (0->2, 2->0, 1保持不变)
                if self.data.iloc[idx]['sentiment'] == 0:
                    self.data.at[idx, 'sentiment'] = 2
                elif self.data.iloc[idx]['sentiment'] == 2:
                    self.data.at[idx, 'sentiment'] = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 处理文本
        text = item['text']
        entity = item['entity']
        
        # 合并文本与实体
        full_text = f"{text} [SEP] {entity}"
        
        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer(
            full_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 处理图像
        image_id = item['image_id']
        # 如果image_id已经包含.jpg扩展名，就直接使用；否则添加.jpg
        if not image_id.endswith('.jpg'):
            image_id = f"{image_id}.jpg"
        image_path = os.path.join(self.image_path, image_id)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # 如果图像加载失败，使用随机噪声图像
            print(f"Error loading image {image_path}: {e}")
            image = torch.randn(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        # 获取标签
        label = torch.tensor(item['sentiment'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': label,
            'text': text,
            'entity': entity,
            'image_id': image_id
        }

def get_data_loaders(config, dataset_name='twitter2015'):
    """
    创建训练、验证和测试数据加载器
    Args:
        config: 配置对象
        dataset_name: 数据集名称
    Returns:
        train_loader, val_loader, test_loader, noise_test_loader
    """
    # 创建数据集
    train_dataset = TwitterDataset(config, dataset_name, 'train')
    val_dataset = TwitterDataset(config, dataset_name, 'dev')
    test_dataset = TwitterDataset(config, dataset_name, 'test')
    noise_test_dataset = TwitterDataset(config, dataset_name, 'test', add_noise=True)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    noise_test_loader = DataLoader(
        noise_test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, noise_test_loader 