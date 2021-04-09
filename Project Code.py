
import numpy as np
import random
from math import exp


def sigmoid(x):
    if len([x]) > 1:
        raise ValueError('wrong value in sigmoid function, dimension of x is %s', x.shape)
    else:
        try:
            ret = 1 / (1 + exp(-x))
        except OverflowError:
            ret = 0
        return ret


class LogisticRegreesionClassifier:

    def __init__(self, penalty='l2', epsilon=0.1, maxiter=500, lmbda=10):
        self.penaly = penalty
        self.w = np.array([])
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.lmbda = lmbda
        self.get_params()

    def __str__(self):
        s = "LogisticRegreesion("
        dic = self.get_params()
        for key in dic:
            s = s + str(key) + "=" + str(dic[key]) + ", "
        s = s + ")"
        return s

    def fit(self, x, y):
        n_samples, n_features = x.shape
        # random pick initial weight in (0, 1)
        w = np.array([])
        for i in range(0, n_features+1):
            w = np.append(w, random.randint(1, 10000) / 10000)

        lmbda = self.lmbda
        for k in range(0, self.maxiter):
            for indexi, i in enumerate(x):
                xi = np.append(i, 1)    # x = [x, 1]
                mul = w.dot(xi.transpose())    # w^Tx + b
                error = y[indexi] - sigmoid(mul)    # y - g(w^Tx+b)
                if self.penaly == 'l2':
                    var = self.epsilon * (error * xi - w * (1 / lmbda))
                elif self.penaly == 'l1':
                    var = self.epsilon * (error * xi - np.ones((1, n_features+1)) * (1 / lmbda))
                else:
                    var = self.epsilon * (error * xi)    # w = w + e(x(y - g(w^Tx+b)))
                w = np.add(w, var)
        self.w = w

    def predict(self, x):
        ans = []
        for indexi, i in enumerate(x):
            mul = sigmoid(self.w.dot(np.append(i, 1)))
            if mul >= 0.5:
                ans.append(1)
            else:
                ans.append(0)
        return ans

    def get_params(self):
        ret = dict()
        ret["penalty"] = self.penaly
        ret["epsilon"] = self.epsilon
        ret["maxiter"] = self.maxiter
        ret["lambda"] = self.lmbda
        return ret

try:
    import pandas as pd
    import numpy as np
    from sklearn import metrics, preprocessing, model_selection, svm, neural_network
    from sklearn.decomposition import PCA
    # from sklearn.cross_validation import cross_val_score # K折交叉验证模块
    # from sklearn.model_selection import KFold, cross_val_score
    import logistic as lg
except ImportError or SystemError:
    print("ImportError Occurs")

TRAINING_DATA = "./dataset/training.csv"
VALIDATION_DATA = "./dataset/validation.csv"
TEST_DATA = "./dataset/SPECTF_TEST.csv"


def data_preprocessing():
    print("Data preprocessing...")
    data = pd.read_csv("./dataset/SPECTF_train.csv", header=None, low_memory=False)
    # split to training data and validation data (ratio 1:4)
    train_data, validation_data = model_selection.train_test_split(data, test_size=0.2, random_state=1)
    train_data.to_csv("./dataset/training.csv", index=False)
    validation_data.to_csv("./dataset/validation.csv", index=False)


def read_data(data_path):
    data = pd.read_csv(data_path, header=None, low_memory=False)
    column_list = data.columns.values  # header of data
    x = data[column_list[1:]].values  # features
    y = data[column_list[0]].values  # class labels
    return x, y


def dimension_reduction(train_x, validation_x, test_x, p=0.9):
    print("Processing PCA...")
    pca = PCA().fit(train_x)
    cnt = 0
    p_sum = 0
    for i in pca.explained_variance_ratio_:
        cnt += 1
        p_sum += i
        if p_sum >= p:
            break

    pca = PCA(n_components=cnt).fit(train_x)
    train_x = pca.transform(train_x)
    validation_x = pca.transform(validation_x)
    test_x = pca.transform(test_x)
    return train_x, validation_x, test_x


def model_fit(train_x, train_y, validation_x, validation_y, test_x, test_y, clf):
    # fit model
    clf.fit(train_x, train_y)
    # generate predict label of validation dataset
    v_res = clf.predict(validation_x)
    # calculate accuracy for validation label
    v_score = metrics.accuracy_score(validation_y, v_res)
    # num of correct labels of validation dataset
    v_correct_label_num = metrics.accuracy_score(validation_y, v_res, normalize=False)
    v_total_num = len(validation_y)

    # generate predict label of test data
    t_res = clf.predict(test_x)
    # calculate accuracy for test label
    t_score = metrics.accuracy_score(test_y, t_res)
    # num of correct labels of test dataset
    t_correct_label_num = metrics.accuracy_score(test_y, t_res, normalize=False)
    t_total_num = len(test_y)
    return v_score, t_score, v_correct_label_num, t_correct_label_num, v_total_num, t_total_num


def main():
    # different classifiy methods
    clf_list = [
        lg.LogisticRegreesionClassifier(penalty="l1", epsilon=0.1, maxiter=10, lmbda=10),
        svm.SVC(kernel="linear", probability=True, C=3),
        neural_network.MLPClassifier(max_iter=500, activation='logistic', hidden_layer_sizes = (20,3), alpha=0.1),
    ]

    print("starts")
    # prepare data
    data_preprocessing()
    # read in data
    train_x, train_y = read_data(TRAINING_DATA)
    # train_x, train_y = read_data("./dataset/SPECTF_train.csv")
    validation_x, validation_y = read_data(VALIDATION_DATA)
    test_x, test_y = read_data(TEST_DATA)
    # dimension reduction
    train_x, validation_x, test_x = dimension_reduction(train_x, validation_x, test_x, p=0.99)



    for clf in clf_list:
        #print("Running ", clf.__str__())  # print name of classifier
        v_score, t_score, v_correct_label_num, t_correct_label_num, v_total_num, t_total_num = model_fit(train_x,
                                                                                                         train_y,
                                                                                                         validation_x,
                                                                                                         validation_y,
                                                                                                         test_x, test_y,
                                                                                                         clf)
        # result
        print(" ")
        print("Validation Accuracy=", v_score)
        print("Test Accuracy=", t_score)
        print("------")

        # scores = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy')
        # print('scores', scores)
        # print('acc', scores.mean())


    return


if __name__ == '__main__':
    main()
