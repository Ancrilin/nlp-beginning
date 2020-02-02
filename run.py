from utils import build_dataset, train_val_test_split, build_iterator, get_pre_embedding_weight
# from TextClassify.LogisticRegressionClassifier import Model
import numpy as np
from TextClassify.SoftmaxRegressionClassifier import Model
from TextClassify.TextCNN import Config, Model
from train_eval import train, predict
import torch
import pandas as pd

if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)                        # 设置随机种子用来保证模型初始化的参数是一致
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True   # 保证每次结果一样, 因为计算有随机性，每次前馈结果略有差异
    config = Config('pred')
    contents, p_contents, vocab = build_dataset(config)
    print('word2vec_path: ', config.word2vec)
    config.pred_embedding_weights, config.embed = get_pre_embedding_weight(config.word2vec, vocab, config)
    print('n_dimension', config.embed)
    print('config.pred_embedding_weights size', config.pred_embedding_weights.size())
    # print('config.pred_embedding_weights', config.pred_embedding_weights)
    print('len(vocab)', len(vocab))
    config.n_vocab = len(vocab)
    contents = np.array(contents)
    p_contents = np.array(p_contents)
    print('len contents', len(contents))
    print('len p_contents', len(p_contents))
    print('vocab', vocab)
    model = Model(config).to(config.device)
    y = np.ones(len(contents))                                                         # 临时使用
    train_contents, val_contents, test_contents, y_train, y_val, y_test = train_val_test_split(contents, y,
                                 val_size=config.VAL_SIZE, test_size=config.TEST_SIZE, shuffle=True)
    train_iter = build_iterator(train_contents, config, True)
    dev_iter = build_iterator(val_contents, config, True)
    test_iter = build_iterator(test_contents, config, True)
    data_iter = build_iterator(p_contents, config, False)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    pred = predict(config, model, data_iter)
    print('pred', pred)
    print('np.shape(pred)', np.shape(pred))
    print('pred', pred)
    pred = np.array(pred).squeeze()
    print('np.shape(pred)', np.shape(pred))
    save = pd.DataFrame({'PhraseId': p_contents[:, 2], 'Sentiment': pred})
    save.to_csv('result/' + config.model_name +  '_test_pred.csv', index=None)






