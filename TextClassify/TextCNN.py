import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self):
        self.model_name = 'TextCNN'
        self.dataset = 'data/Sentiment Analysis on Movie Reviews'
        self.TRAIN_SIZE = 0.7
        self.VAL_SIZE = 0.15
        self.TEST_SIZE = 0.15
        self.channel_size = 1
        self.n_cluster = 5
        self.min_freq = 1
        self.MAX_VOCAB_SIZE = 20000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 16                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.embed = 300                                                # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.channel_size, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

