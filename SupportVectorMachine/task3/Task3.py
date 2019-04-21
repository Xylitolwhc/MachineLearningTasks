# -*- coding: utf-8 -*-
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import time


def loadData(filename):
    """
    输入:
        数据集路径
    输出:
        numpy.array格式的X, y数据array
        X为m×n的数据array, m为样例数, n为特征维度
        y为m×1的标签array, 1表示正例, 0表示反例
    """
    dataDict = loadmat(filename)
    return dataDict['X'], dataDict['y']


def plotData(X, y, title=None):
    """
    作出原始数据的散点图
    X, y为loadData()函数返回的结果
    """

    X_pos = []
    X_neg = []

    sampleArray = np.concatenate((X, y), axis=1)
    for array in list(sampleArray):
        if array[-1]:
            X_pos.append(array)
        else:
            X_neg.append(array)

    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    if title: ax.set_title(title)

    pos = plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='+', c='b')
    neg = plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='o', c='y')

    plt.legend((pos, neg), ('postive', 'negtive'), loc=2)

    plt.show()


def svmTrain_SMO(X, y, C, kernelFunction='linear', tol=1e-3, max_iter=5, **kargs):
    """
    利用简化版的SMO算法训练SVM
    （参考《机器学习实战》）

    输入：
    X, y为loadData函数的返回值
    C为惩罚系数
    kernelFunction表示核函数类型, 对于非线性核函数，也可直接输入核函数矩阵K
    tol为容错率
    max_iter为最大迭代次数

    输出：
    model['kernelFunction']为核函数类型
    model['X']为支持向量
    model['y']为对应的标签
    model['alpha']为对应的拉格朗日参数
    model['w'], model['b']为模型参数
    """

    m, n = X.shape
    X = np.mat(X)
    y = np.mat(y, dtype='float64')

    y[np.where(y == 0)] = -1

    alphas = np.mat(np.zeros((m, 1)))
    b = 0.0
    E = np.mat(np.zeros((m, 1)))
    iters = 0
    eta = 0.0
    L = 0.0
    H = 0.0

    if kernelFunction == 'linear':
        K = X * X.T
    elif kernelFunction == 'gaussian':
        K = kargs['K_matrix']
    else:
        print('Kernel Error')
        return None

    while iters < max_iter:

        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(np.multiply(np.multiply(alphas, y), K[:, i])) - y[i]

            if (y[i] * E[i] < -tol and alphas[i] < C) or (y[i] * E[i] > tol and alphas[i] > 0):
                j = np.random.randint(m)
                while j == i:
                    j = np.random.randint(m)

                E[j] = b + np.sum(np.multiply(np.multiply(alphas, y), K[:, j])) - y[j]

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta

                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - E[i] \
                     - y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - y[j] * (alphas[j] - alpha_j_old) * K[i, j]

                b2 = b - E[j] \
                     - y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                     - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if (0 < alphas[i] and alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] and alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            iters = iters + 1
        else:
            iters = 0

    idx = np.where(alphas > 0)
    model = {'X': X[idx[0], :], 'y': y[idx], 'kernelFunction': str(kernelFunction), \
             'b': b, 'alphas': alphas[idx], 'w': (np.multiply(alphas, y).T * X).T}
    return model


def svmPredict(model, X, *arg):
    """
    利用得到的model, 计算给定X的模型预测值

    输入：
    model为svmTrain_SMO返回值
    X为待预测数据
    sigma为训练参数
    """

    m = X.shape[0]
    p = np.mat(np.zeros((m, 1)))
    pred = np.mat(np.zeros((m, 1)))

    p = X * model['w'] + model['b']

    pred[np.where(p >= 0)] = 1
    pred[np.where(p < 0)] = 0

    return pred


def accuracy(predict, y):
    wrong_num = 0.0
    for i in range(len(predict)):
        if predict[i] != y[i]:
            wrong_num += 1
    return 1.0 - float(wrong_num) / float(len(y))


def __main__():
    start = time.clock()

    # Load Train Data
    train_data, train_labels = loadData("./task3_train.mat")
    print("Train Data Dimension:\t", len(train_data[0]))
    print("Train Data Quantity:\t", len(train_data))

    # lda = LinearDiscriminantAnalysis(n_components=1000)
    # lda.fit(X=train_data, y=train_labels.ravel())
    # train_data = lda.transform(X=train_data)

    # Train Model
    # C = 1
    # sigma = 0.01
    # linear_model = svmTrain_SMO(X=train_data, y=train_labels, C=C, kernelFunction='linear',
    #                             tol=1e-5, max_iter=2)
    # predict_y = svmPredict(linear_model, train_data, sigma)
    # print("%10s" % ("%.5f" % accuracy(predict_y, train_labels)))

    # Predict
    test_data = loadmat("./task3_test.mat")['X']
    # test_labels = svmPredict(linear_model, test_data, sigma)

    # parameters = {"C": [1, 10], "gamma": [1, 0.1, 0.01], "kernel": ["linear"]}
    clf = svm.SVC(C=1, gamma=1, kernel="linear")
    # cv = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    # cv.fit(train_data, train_labels.ravel())
    # print(cv.best_estimator_)
    # print(cv.best_params_)
    # print(cv.best_score_)

    test_labels = clf.fit(train_data, train_labels.ravel()).predict(test_data)
    print("Test Data Dimension:\t", len(test_data[0]))
    print("Test Data Quantity:\t", len(test_data))
    with open("test_labels.txt", "w") as file:
        for label in list(test_labels):
            file.write(str(label))
            file.write("\n")

    end = time.clock()
    print()
    print('Time used:\t', str(end - start), 's')


if __name__ == '__main__':
    __main__()
