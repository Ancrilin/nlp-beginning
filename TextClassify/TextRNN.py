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
        self.embed = 50                                                 # 字向量维度


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pred_embedding_weights is None:
            print('random embedding')
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        else:
            print('pred embedding')
            self.embedding = nn.Embedding.from_pretrained(config.pred_embedding_weights, freeze=False)
        # RNN中：batchsize的默认位置是position 1

