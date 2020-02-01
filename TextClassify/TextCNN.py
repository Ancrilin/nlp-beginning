import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self, embedding):
        self.model_name = 'TextCNN'
        self.embedding = embedding
        self.dataset = 'data/Sentiment Analysis on Movie Reviews'
        self.word2vec = 'data/glove/glove.42B.300d/glove.42B.300d.word2vec.txt'
        self.save_path = 'result/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path =  'result/log/' + self.model_name
        self.pred_embedding_weights = None
        self.TRAIN_SIZE = 0.7
        self.VAL_SIZE = 0.15
        self.TEST_SIZE = 0.15
        self.channel_size = 1
        self.n_cluster = 5
        self.min_freq = 1
        self.MAX_VOCAB_SIZE = 20000
        self.num_classes = 5
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.class_list = [x.strip() for x in open(
            'data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 16                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 16                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.embed = 50                                                 # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pred_embedding_weights is None:
            print('random embedding')
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        else:
            print('pred embedding')
            self.embedding = nn.Embedding.from_pretrained(config.pred_embedding_weights, freeze=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.channel_size, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # (batch, sentence_length)
        out = self.embedding(x[0])
        # (batch, sentence_length, embed_dim)
        out = out.unsqueeze(1)                #重新拼接, 等价于out=out.view(out.size(0),1,max_len,word_dim)
        # (batch, 1, sentence_length, embed_dim)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  #3个卷积拼接成一个长向量
        # (batch, kernel_num)
        # (batch, 3 * kernel_num)
        out = self.dropout(out)
        out = self.fc(out)
        return out