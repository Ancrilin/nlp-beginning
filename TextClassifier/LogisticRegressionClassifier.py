import numpy as np
from tqdm import tqdm


class Model():
    def __init__(self, max_iter = 300, learning_rate = 0.01):
        self.max_iter = max_iter                                                #最大迭代次数
        self.learning_rate = learning_rate                                      #x学习步长

    #对数似然函数求极大值
    def fit(self, X, y):
        print('fitting...')
        X = self._append_bias(X)
        self.weights = np.zeros((len(X[0]), 1), dtype=np.float32)
        for iter_ in tqdm(range(self.max_iter)):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(X[i], self.weights))
                error = y[i] - result
                # 梯度下降迭代权重参数self.weights
                self.weights += self.learning_rate * error * np.transpose([X[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def evaluate(self, X_test, y_test):
        X_test = self._append_bias(X_test)
        y_pred = []
        right = 0
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if(result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)

    def _append_bias(self, X):
        data = []
        for line in X:
            data.append([*line, 1.0])
        return np.array(data, dtype=np.float)

if  __name__ == '__main__':
    X = [[1,2,3], [4,5,6], [7,8,9]]
    t = []
    y = [1,0,1]
    for d in X:
        t.append([1., *d])
    print(t)
    print(np.zeros((len(X[0]),1), dtype=np.float))
    model = Model()
    model.fit(X, y)
    model.evaluate(X, y)

