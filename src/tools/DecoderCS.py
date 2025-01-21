"""
用于对神经表征进行识别，以检查神经表征是否提供了某种分类信息
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn import preprocessing
import numpy as np


class BinaryClassifier:
    """
    将线性二分类器的训练、测试和预测，集成在一个类中
    """

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        # self.model = LogisticRegression(max_iter=2000)  # 创建线性二分类器
        self.model = LogisticRegression()  # 创建线性二分类器

    def dataSetting(self, X_train, y_train, X_test, y_test):
        """
        :param X: (sampleNum, dim), 每一行是一个样本，最好第一列为全1
        :param y: (sampleNum)
        :return: None
        """

        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        # 正规化
        self.scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        # self.model.fit(np.array(X_scaled), self.y_train)
        self.model.fit(np.array(X_scaled) + 3*np.random.randn(X_scaled.shape[0], X_scaled.shape[1]), self.y_train)

    def dataLoadAndTrain(self, X, y, test_size=0.2):
        """

        :param X: (sampleNum, dim), 每一行是一个样本，最好第一列为全1
        :param y: (sampleNum)
        :param test_size:
        :return: None
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)
        self.dataSetting(self.X_train, self.y_train, self.X_test, self.y_test)

    def dataTest(self):
        # 在测试集上进行测试
        y_pred = self.model.predict(self.scaler.transform(self.X_test))
        # 计算并输出准确率
        accuracy = accuracy_score(self.y_test, y_pred)
        # print(f"Accuracy: {accuracy}")
        return accuracy

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


if __name__ == '__main__':
    import numpy as np

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    # X = np.random.rand(10, 3)
    # y = np.array(np.random.rand(10) > 0.5).astype(float)
    classifier = BinaryClassifier()
    classifier.dataLoadAndTrain(X, y)
    print(classifier.dataTest())
