import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from utils.util import data_processor, get_time_dif, get_pre_embedding
from utils.dataset import MyDataset
from TextClassifier.TextESIM import Config, Model
import time
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report


def train(model, train_dataset, dev_dataset, config):
    start_time = time.time()
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    n_sample = len(train_dataloader)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    classified_loss = torch.nn.CrossEntropyLoss().to(config.device)
    total_batch = 0
    for epoch in range(config.num_epochs):
        total_loss = 0
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for sample in tqdm(train_dataloader):
            sample = (i.to(config.device) for i in sample)
            sentence1, sentence2, label, seq1, seq2 = sample
            # print(sentence1)
            model.zero_grad()
            out = model(sentence1, sentence2)
            loss = classified_loss(out, label)
            total_loss += loss.clone().detach()
            loss.backward()
            optimizer.step()
            total_batch += 1
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        print(f'train loss: {total_loss / n_sample}')
        evaluate(config, model, dev_dataset)
        torch.save(model.state_dict(), config.save_path)


def evaluate(config, model, dataset):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    n_sample = len(dataloader)
    # print('len dataloader', len(dataset))
    with torch.no_grad():
        for sample in tqdm(dataloader):
            sample = (i.to(config.device) for i in sample)
            sentence1, sentence2, label, seq1, seq2 = sample
            out = model(sentence1, sentence2)
            loss = F.cross_entropy(out, label)
            loss_total += loss
            label = label.data.cpu().numpy()
            predic = torch.max(out.data, 1)[1].cpu().numpy()                # [0]为值, [1]为索引
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)
    report = classification_report(labels_all, predict_all, labels=[0, 1, 2],
                                   target_names=['entailment', 'neutral', 'contradiction'])
    print('---------------------------------------')
    print(f'evaluate loss: {loss_total / n_sample}')
    print(report)
    return labels_all, predict_all


def test(model, config, test_dataset):
    labels_all, predict_all = evaluate(config, model, test_dataset)
    accuracy = 0
    for i in range(len(labels_all)):
        if labels_all[i] == predict_all[i]:
            accuracy += 1
    print('accuracy: ', accuracy / len(labels_all))
    return labels_all, predict_all

def load_model(model, save_path):
    model = model.load_state_dict(torch.load(save_path))



if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)                                                 # 设置随机种子用来保证模型初始化的参数是一致
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True                            # 保证每次结果一样, 因为计算有随机性，每次前馈结果略有差异
    config = Config('glove')                                            # random, glove
    print('device', config.device)
    vocab = None
    train_dataset, vocab = data_processor(config.train_path, config, vocab)
    dev_dataset, vocab = data_processor(config.dev_path, config, vocab)
    test_dataset, vocab = data_processor(config.test_path, config, vocab)
    config.n_vocab = len(vocab)
    if config.embedding != 'random':
        print('loading pre embedding...')
        weight, vector_size = get_pre_embedding(vocab, config.pred_embedding_path)
        config.pred_embedding_weights = weight
        config.embed = vector_size
    train_dataset = MyDataset(train_dataset, config)
    dev_dataset = MyDataset(dev_dataset, config)
    # print(vocab)
    model = Model(config).to(config.device)
    print(model)
    train(model, train_dataset, dev_dataset, config)
    print('test...')
    test_dataset = MyDataset(test_dataset, config)
    labels_all, predict_all = test(model, config, test_dataset)
