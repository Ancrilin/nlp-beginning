import pandas as pd
import torch
import numpy as np
from utils import build_dataset, train_val_test_split, get_pre_embedding_weight
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from TextClassify.TextCNN import Config
import torch.nn.functional as F


if __name__ == '__main__':
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
    # TRAIN_SIZE = 0.7
    # VAL_SIZE = 0.15
    # TEST_SIZE = 0.15
    # dataset = 'data/Sentiment Analysis on Movie Reviews'
    # train, test_content = build_dataset(dataset, 16, 1, 20000)
    # train = np.array(train)
    # print(train[0])
    # test_content = np.array(test_content)
    # print(test_content[0])
    # print(test_content[:, 0])
    # print(test_content[:, 3])
    # save = pd.DataFrame({'PhraseId': test_content[:, 3], 'Sentiment': train[:len(test_content), 2]})
    # save.to_csv('result.csv')
    config = Config('pred')
    contents, p_contents, vocab = build_dataset(config)
    print(vocab)
    config.n_vocab = len(vocab)
    print(config.n_vocab)
    glove_input_file = 'data/glove/glove.6B/glove.6B.300d.txt'
    word2vec_output_file = 'data/glove/glove.6B/glove.6B.50d.word2vec.txt'
    print(word2vec_output_file)
    weights = get_pre_embedding_weight(word2vec_output_file, vocab, config)
    print(weights.size(), weights)
    emb = (F.embedding(torch.tensor([[1,2], [3,4]]), weights))
    print(emb)
    print(emb.size())