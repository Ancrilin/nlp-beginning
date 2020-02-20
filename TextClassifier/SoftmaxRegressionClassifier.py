import numpy as np
from tqdm import tqdm
from sklearn import preprocessing


class Model():
    def __init__(self, n_cluster, max_iter=300, learning_rate=0.001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # X = self._append_bias(X)
        numSamples, numFeatures = np.shape(X)
        self.weights = np.mat(np.ones((numFeatures, self.n_cluster)), dtype=np.double)
        print(f'numSamples {numSamples}, numFeature {numFeatures}')
        for i in tqdm(range(self.max_iter)):
            left = 0
            right = 32
            if right >= numSamples:
                right = numSamples
            while right < numSamples:
                batch = X[left:right]
                # print(f'i {i} left {left}, right {right}')
                dot = np.dot(batch, self.weights)
                max_dot = np.exp(dot.max(axis=1))
                value = np.exp(dot) / max_dot
                sum = value.sum(axis=1)
                sum = np.repeat(sum, self.n_cluster, axis=1)  # 横向复制
                pro = - value / sum
                for j in range(len(batch)):
                    pro[j, y[j]] += 1
                self.weights += self.learning_rate * np.dot(batch.T, pro)
                left = right
                right += 32
                if right >= numSamples:
                    right = numSamples

    def _append_bias(self, X):
        data = []
        for line in X:
            data.append([*line, 1.0])
        return np.array(data, dtype=np.float)

    def Z_score(self, X):
        X = self._append_bias(X)
        mean = np.mean(X, axis=0)
        print('mean', mean)
        X -= mean
        std = np.std(X, axis=0)
        print('std', std)
        X /= std
        return X

    def stand(self, X):
        X = self._append_bias(X)
        std = preprocessing.StandardScaler()
        X = std.fit_transform(X)
        return X

    def evaluate(self, X_test, y_test):
        # X_test = self._append_bias(X_test)
        result = np.dot(X_test, self.weights)
        y_pred = result.argmax(axis=1)
        count = 0
        for i in range(np.shape(y_pred)[0]):
            if y_pred[i,] == y_test[i,]:
                count += 1
        return count / len(y_test), y_pred

    def predict(self, X):
        result = np.dot(X, self.weights)
        y_pred = result.argmax(axis=1)
        return y_pred


if __name__ == '__main__':
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 1, 6], [5, 5, 1], [9, 2, 5]]
    y = [1, 0, 2, 1, 0, 2]
    model = Model(3, max_iter=3)
    X = model.stand(X)
    # print(X)
    # X = model.Z_score(X)
    model.fit(X, y)
