import torch
import torch.nn as nn
import torch.nn.functional as F


class Config():
    def __init__(self, embedding):
        self.model_name = 'TextESIM'
        self.embedding = embedding
        self.dataset = 'data/snli_1.0'
        self.train_path = self.dataset + '/snli_1.0_train.jsonl'
        self.dev_path = self.dataset + '/snli_1.0_dev.jsonl'
        self.test_path = self.dataset + '/snli_1.0_test.jsonl'
        self.stop_path = 'data/stop.txt'
        self.save_path = 'result/' + self.model_name + '.ckpt' if self.embedding == 'random' else 'result/' + self.model_name + '_pre_embedding.ckpt'
        self.pred_embedding_path = 'data/glove/glove.6B/glove.6B.300d.word2vec.txt'
        self.pred_embedding_weights = None
        self.min_freq = 1                                               # 最小词频
        self.MAX_VOCAB_SIZE = 20000
        self.num_classes = 3                                            # 类别数
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.class_list = [x.strip() for x in open(
            'data/class.txt', encoding='utf-8').readlines()]            # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 4                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0004                                       # 学习率
        self.embed = 300                                                 # 字向量维度
        self.hidden_size_1 = 256                                          # 隐藏层维度
        self.hidden_size_2 = 512


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pred_embedding_weights is None:
            self.embedding = nn.Embedding(config.n_vocab, config.embed)
        else:
            self.embedding = nn.Embedding.from_pretrained(config.pred_embedding_weights, freeze=False)
        self.lstm1 = nn.LSTM(config.embed, config.hidden_size_1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * 4 * config.hidden_size_1, config.hidden_size_2,
                             batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 2 * config.hidden_size_2, config.num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, sentence1, sentence2):
        # [batch, seq]
        sentence1 = self.embedding(sentence1)
        sentence2 = self.embedding(sentence2)
        # [batch, seq, dim]
        out1, _ = self.lstm1(sentence1)
        out2, _ = self.lstm1(sentence2)
        # [batch, seq, 2 * hidden_size_1]
        out1_align, out2_align = self.soft_attention_align(out1, out2)
        m1 = torch.cat([out1, out1_align, out1 - out1_align,
                        out1 * out1_align], dim=-1)                          # [a, a_hat, a - a_hat, a * a_hat]
        m2 = torch.cat([out2, out2_align, out2 - out2_align,
                        out2 * out2_align], dim=-1)                          # element-wise
        # [batch, seq, 4 * 2 * hidden_size_1]
        out1, _ = self.lstm2(m1)
        out2, _ = self.lstm2(m2)
        # [batch, seq, 2 * hidden_size_2]
        # 每句取最大值和均值，除去句子长度的影响
        out = torch.cat([F.avg_pool1d(out1.transpose(1, 2), out1.size(1)).squeeze(-1),
                         F.max_pool1d(out1.transpose(1, 2), out1.size(1)).squeeze(-1),
                         F.avg_pool1d(out2.transpose(1, 2), out2.size(1)).squeeze(-1),
                         F.max_pool1d(out2.transpose(1, 2), out2.size(1)).squeeze(-1),
                         ], dim=-1)
        # [batch, 4 * 2 * hidden_size_2]
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def soft_attention_align(self, x1, x2):
        # x1_x2
        attention = torch.matmul(x1, x2.transpose(1, 2))
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        # x2_x1
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align


