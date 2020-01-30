import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from TextClassify.TextCNN import Config, Model


MAX_VOCAB_SIZE = 20000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

stop = stopwords.words('english')
with open('data/stop.txt', 'r', encoding='utf-8')as stp:  # 停用词表
    for row in stp:
        stop.append(row.strip())  # 去停用词
stop_words = set(stop)

def build_dataset(config):
    train_pd = pd.read_csv(config.dataset + '/train.tsv', sep='\t', header=0)
    test_pd = pd.read_csv(config.dataset + '/test.tsv', sep='\t', header=0)
    data = train_pd[['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']].values
    test = test_pd[['PhraseId', 'SentenceId', 'Phrase']].values
    vocab = {}
    t = []
    print('loading vocab...')
    for each in tqdm(data):
        t_line = each[2].lower().split(' ')
        line = []
        for word in t_line:
            if word not in stop_words:
                line.append(word)
                vocab[word] = vocab.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab.items() if _[1] >= config.min_freq], key=lambda x: x[1], reverse=True)[:config.MAX_VOCAB_SIZE]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    vocab = vocab_dic
    contents = []
    print('loading data...')
    for each in tqdm(data):
        t_line = each[2].lower().split(' ')
        line = []
        for word in t_line:
            if word not in stop_words:
                line.append(word)
        seq_len = len(line)
        if config.pad_size:
            if len(line) < config.pad_size:
                line.extend(['<PAD>'] * (config.pad_size - len(line)))
            else:
                line = line[:config.pad_size]
                seq_len = config.pad_size
        words_line = []
        for word in line:
            words_line.append(vocab.get(word, vocab.get(UNK)))  # 根据词典进行转换
        contents.append((words_line, each[3], seq_len, each[0]))
    test_contents = []
    for each in test:
        t_line = each[2].lower().split(' ')
        line = []
        for word in t_line:
            if word not in stop_words:
                line.append(word)
        seq_len = len(line)
        if config.pad_size:
            if len(line) < config.pad_size:
                line.extend(['<PAD>'] * (config.pad_size - len(line)))
            else:
                line = line[:config.pad_size]
                seq_len = config.pad_size
        words_line = []
        for word in line:
            words_line.append(vocab.get(word, vocab.get(UNK)))  # 根据词典进行转换
        test_contents.append((words_line, seq_len, each[0]))
    # print('vocab', vocab)
    return contents, test_contents, vocab

def train_val_test_split(X, y, val_size, test_size, shuffle):
    """Split data into train/val/test datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

class DatasetIterater():
    def __init__(self, batches, batch_size, device):
        self.batches = batches
        self.batch_size = batch_size
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    config = Config()
    contents, p_contents, vocab = build_dataset(config)
    contents = np.array(contents)
    p_contents = np.array(p_contents)
    print('len contents', len(contents))
    print('len p_contents', len(p_contents))
    print('vocab', vocab)
    y = np.ones(len(contents))
    print(y)
    train_contents, val_contents, test_contents, y_train, y_val, y_test = train_val_test_split(contents, y,
                        val_size=config.VAL_SIZE, test_size=config.TEST_SIZE, shuffle=True)
    train_iter = build_iterator(train_contents, config)
    print('train_contents[0]', train_contents[0])
    for i, (trains, labels) in enumerate(train_iter):
        print('i', i)
        print('trains', trains)
        print('labels', labels)
        break




