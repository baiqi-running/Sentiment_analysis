import os
import torch

class Config:
    # 数据集路径
    DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
    TWITTER15_PATH = os.path.join(DATASET_PATH, 'twitter2015')
    TWITTER15_IMAGES_PATH = os.path.join(DATASET_PATH, 'twitter2015_images')
    TWITTER17_PATH = os.path.join(DATASET_PATH, 'twitter2017')
    TWITTER17_IMAGES_PATH = os.path.join(DATASET_PATH, 'twitter2017_images')
    
    # 数据预处理参数
    MAX_TEXT_LENGTH = 128  # BERT输入文本长度
    IMAGE_SIZE = 224  # 图像大小
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    NOISE_TEST_SAMPLES = 100  # 噪声测试子集样本数
    
    # 模型参数
    BERT_MODEL_NAME = 'bert-base-uncased'
    VISION_MODEL_NAME = 'resnet50'
    NUM_CLASSES = 3  # 负面、中性、正面
    DROPOUT_RATE = 0.1
    HIDDEN_SIZE = 768
    IMAGE_EMBED_DIM = 2048
    FUSION_DIM = 512
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    EARLY_STOP_PATIENCE = 5
    LABEL_SMOOTHING = 0.1
    
    # 对比学习参数
    TEMPERATURE = 0.07
    CONTRASTIVE_WEIGHT = 0.3
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出路径
    OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True) 