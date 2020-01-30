from utils import build_dataset, train_val_test_split
# from TextClassify.LogisticRegressionClassifier import Model
import numpy as np
from TextClassify.SoftmaxRegressionClassifier import Model
import pandas as pd
from TextClassify.TextCNN import Config, Model

if __name__ == '__main__':
    config = Config()
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    max_iter = 2000
    n_cluster = 5
    pad_size = 16
    dataset = 'data/Sentiment Analysis on Movie Reviews'
    train, test_content = build_dataset(dataset, pad_size, 1, 20000)
    train = np.array(train)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=train[:, 0], y=train[:, 2], val_size=VAL_SIZE, test_size=TEST_SIZE, shuffle=True)
    model = Model(n_cluster, max_iter=max_iter)                                            #softmax5个类别
    X_train = model.stand(X_train)
    print(X_train)
    model.fit(X_train, y_train)
    X_test = model.stand(X_test)
    result = model.evaluate(X_test, y_test)
    accuracy, y_pred = model.evaluate(X_test, y_test)
    print('y_pred')
    print(y_pred)
    print('y_test')
    print(y_test)
    print('accuracy', accuracy)
    test_content = np.array(test_content)
    test_content_std = model.stand(test_content[:, 0])
    test_content_result = model.predict(test_content_std)
    print('test content pid', test_content[:, 3])
    test_content_result = np.array(test_content_result).squeeze()
    print('test result')
    print(test_content_result)
    save = pd.DataFrame({'PhraseId':test_content[:, 3], 'Sentiment':test_content_result})
    save.to_csv('result_iter_' + str(max_iter) + '.csv', index=None)




