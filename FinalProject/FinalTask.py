import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import copy
import math
import random


def accuracy(original_x, predict_x, original_y, predict_y):
    if len(original_y) != len(predict_y) \
            or len(original_x) != len(predict_x) \
            or len(original_x) != len(original_y):
        return
    return np.average(np.sqrt(np.add(
        np.power(np.subtract(original_y, predict_y), 2),
        np.power(np.subtract(original_x, predict_x), 2)
    )))


# 两层模型算法训练函数
def split_train(sector_all, x, y):
    block_size = 4  # 每个分块正方形的边长，单位m
    block_num = int(64 * 32 / (block_size * block_size))  # 所分块的个数
    model1 = RandomForestClassifier(n_estimators=200)  # 第一层模型使用随机森林分类器
    model2 = {}  # 第二次模型使用字典存储
    split_data = {}
    for i in range(block_num):
        split_data[i] = [[], [], []]  # 将训练集分块后存储，0位置存储样本特征向量，1位置存储x值，2位置存储y值
        model2[i] = []
    labels = []
    for i in range(len(x)):
        label = int(x[i] / block_size) + int(y[i] / block_size) * int(32 / block_size)
        labels.append(label)
        split_data[label][0].append(sector_all[i])
        split_data[label][1].append(x[i])
        split_data[label][2].append(y[i])
    model1 = model1.fit(sector_all, labels)  # 训练第一层模型
    for label in split_data:
        model2_x = RandomForestRegressor(n_estimators=200)  # 第二层模型使用随机森林回归算法，分别预测x,y
        model2_y = RandomForestRegressor(n_estimators=200)
        if len(split_data[label][0]) != 0 and len(split_data[label][1]) != 0 and len(split_data[label][2]) != 0:
            model2_x = model2_x.fit(split_data[label][0], split_data[label][1])
            model2_y = model2_y.fit(split_data[label][0], split_data[label][2])
        model2[label] = [model2_x, model2_y]
    return model1, model2


def split_predict(model1, model2, sector_all):
    labels = model1.predict(sector_all)
    predict_x = []
    predict_y = []
    for i in range(len(labels)):
        if len(model2[labels[i]]) != 0:
            model2_x = model2[labels[i]][0]
            model2_y = model2[labels[i]][1]
            predict_x.append(model2_x.predict([sector_all[i]])[0])
            predict_y.append(model2_y.predict([sector_all[i]])[0])
        else:
            print("No Model")
            predict_x.append(0)
            predict_y.append(0)
    return predict_x, predict_y


def test(model, features, x, y):
    acc = 0.0
    for i in range(5):
        original_x, predict_x = train(copy.deepcopy(model), features, x)
        original_y, predict_y = train(copy.deepcopy(model), features, y)
        acc += accuracy(original_x, predict_x, original_y, predict_y)
    acc /= 5.0
    print("%10s" % ("%.5f" % acc), "m", sep='')
    return


def train(model, x, y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=None)
    model = model.fit(X_train, Y_train)
    return Y_test, model.predict(X_test)


def model_test(features, x, y):
    model = LinearRegression()
    print("%-30s" % "LinearRegression:", end="")
    test(model, features, x, y)

    model = Lasso()
    print("%-30s" % "Lasso:", end="")
    test(model, features, x, y)

    model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(50, 50, 50, 50), tol=1e-5, max_iter=500)
    print("%-30s" % "MLPRegressor:", end="")
    test(model, features, x, y)

    model = svm.SVR(C=100, tol=1e-5, gamma='scale')
    print("%-30s" % "SVR:", end="")
    test(model, features, x, y)

    model = neighbors.KNeighborsRegressor()
    print("%-30s" % "KNeighborsRegressor:", end="")
    test(model, features, x, y)

    model = tree.DecisionTreeRegressor()
    print("%-30s" % "DecisionTreeRegressor:", end="")
    test(model, features, x, y)

    model = RandomForestRegressor(n_estimators=200)
    print("%-30s" % "RandomForestRegressor:", end="")
    test(model, features, x, y)

    model = AdaBoostRegressor(n_estimators=200)
    print("%-30s" % "AdaBoostRegressor:", end="")
    test(model, features, x, y)

    model = GradientBoostingRegressor(n_estimators=400, min_samples_split=3)
    print("%-30s" % "GradientBoostingRegressor:", end="")
    test(model, features, x, y)


def save_predict(features, x, y):
    test = pd.read_csv("./testAll.csv")
    reg = RandomForestRegressor(n_estimators=200)
    reg = reg.fit(features, x)
    predict_x = reg.predict(test)
    reg = RandomForestRegressor(n_estimators=200)
    reg = reg.fit(features, y)
    predict_y = reg.predict(test)
    with open("./test_labels.csv", "w") as file:
        file.write("id,x,y\n")
        for i in range(len(predict_x)):
            file.write(str(i) + "," + str("%.3f" % predict_x[i]) + "," + str("%.3f" % predict_y[i]) + "\n")


def split_train_test(sector2_1, sector3_5, x_y):
    sector_all = np.column_stack((sector2_1, sector3_5))
    repeat_times = 5
    accuracy = 0.0
    for i in range(repeat_times):
        X_train, X_test, Y_train, Y_test = train_test_split(sector_all, x_y, test_size=0.1, random_state=None)
        model1, model2 = split_train(X_train, list(Y_train['x']), list(Y_train['y']))
        predict_x, predict_y = split_predict(model1, model2, X_test)
        sum = 0.0
        for i in range(len(predict_x)):
            sum += math.sqrt((predict_x[i] - list(Y_test['x'])[i]) ** 2 + (predict_y[i] - list(Y_test['y'])[i]) ** 2)
        # print(predict_x[i], "\t", list(Y_test['x'])[i], "\t", predict_y[i], "\t", list(Y_test['y'])[i])
        accuracy += sum / len(predict_x)
    print(accuracy / repeat_times)
    return accuracy / repeat_times


def part_split_train_test(data, x_y):
    sum = 0
    repeat_time = 10
    for i in range(repeat_time):
        sector2_1, sector3_5 = random_sectors(data, 9)
        sum += split_train_test(sector2_1, sector3_5, x_y)
    print("Avg Accuracy:\t", sum / repeat_time, sep="")


def save_split_predict(sector2_1, sector3_5, x_y):
    test = pd.read_csv("./testAll.csv")
    sector_all = np.column_stack((sector2_1, sector3_5))
    model1, model2 = split_train(sector_all, list(x_y['x']), list(x_y['y']))
    predict_x, predict_y = split_predict(model1, model2, test.values)
    with open("./test_labels.csv", "w") as file:
        file.write("id,x,y\n")
        for i in range(len(predict_x)):
            file.write(str(i) + "," + str("%.3f" % predict_x[i]) + "," + str("%.3f" % predict_y[i]) + "\n")


def random_sectors(data, del_sector_num):
    sector_nums = []
    for i in range(15):
        sector_nums.append(i)
    for i in range(del_sector_num):
        while True:
            r = random.randint(0, 15)
            if r in sector_nums:
                sector_nums.remove(r)
                break
    print(sector_nums)
    columns = []
    for num in sector_nums:
        columns.append("2100" + str(num))
    sector2_1 = data[columns]
    columns = []
    for num in sector_nums:
        columns.append("3500" + str(num))
    sector3_5 = data[columns]
    return sector2_1, sector3_5


def __main__():
    data = pd.read_csv("./dataAll.csv")
    x = data['x']
    y = data['y']
    x_y = data[['x', 'y']]

    sector2_1, sector3_5 = random_sectors(data, 0)

    # features = sector3_5.values
    features = np.column_stack((sector2_1, sector3_5))
    model_test(features, x, y)
    # save_predict(features,x,y)

    split_train_test(sector2_1, sector3_5, x_y)
    # part_split_train_test(data, x_y)
    # save_split_predict(sector2_1, sector3_5, x_y)


if __name__ == '__main__':
    __main__()
