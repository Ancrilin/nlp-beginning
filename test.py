import pandas as pd
import torch
import numpy as np
from utils import build_dataset, train_val_test_split


if __name__ == '__main__':
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
    for k in (2, 3, 4) :
        print(k)
