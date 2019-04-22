import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import copy
import math


def accuracy(original_x, predict_x, original_y, predict_y):
    if len(original_y) != len(predict_y) \
            or len(original_x) != len(predict_x) \
            or len(original_x) != len(original_y):
        return
    return np.average(np.sqrt(np.add(
        np.power(np.subtract(original_y, predict_y), 2),
        np.power(np.subtract(original_x, predict_x), 2)
    )))


def split_train(sector_all, x, y):
    # sector_all = np.column_stack((sector2_1, sector3_5))
    block_size = 4
    block_num = int(64 * 32 / (block_size * block_size))
    model1 = RandomForestClassifier(n_estimators=200)
    model2 = {}
    split_data = {}
    for i in range(block_num):
        split_data[i] = [[], [], []]
        model2[i] = []
    labels = []
    for i in range(len(x)):
        label = int(x[i] / block_size) + int(y[i] / block_size) * int(32 / block_size)
        labels.append(label)
        split_data[label][0].append(sector_all[i])
        split_data[label][1].append(x[i])
        split_data[label][2].append(y[i])
    model1 = model1.fit(sector_all, labels)
    '''
    for i in split_data:
        for line in split_data[i]:
            print(line)
    '''
    for label in split_data:
        model2_x = RandomForestRegressor(n_estimators=200)
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
    original_x, predict_x = train(copy.deepcopy(model), features, x)
    original_y, predict_y = train(copy.deepcopy(model), features, y)
    print("%10s" % ("%.5f" % accuracy(original_x, predict_x, original_y, predict_y)), "m", sep='')
    return


def train(model, x, y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=None)
    model = model.fit(X_train, Y_train)
    return Y_test, model.predict(X_test)


def __main__():
    data = pd.read_csv("./dataAll.csv")
    x = data['x']
    y = data['y']
    x_y = data[['x', 'y']]
    columns = []
    for i in range(15):
        columns.append("2100" + str(i))
    sector2_1 = data[columns]
    columns = []
    for i in range(15):
        columns.append("3500" + str(i))
    sector3_5 = data[columns]

    features = sector3_5
    '''
    model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(50, 50, 50, 50), tol=1e-6, max_iter=1000)
    print("%-25s" % "MLPRegressor:", end="")
    test(model, features, x, y)

    model = LinearRegression()
    print("%-25s" % "LinearRegression:", end="")
    test(model, features, x, y)

    model = svm.SVR(C=100, tol=1e-5, gamma=0.01)
    print("%-25s" % "SVR:", end="")
    test(model, features, x, y)

    model = tree.DecisionTreeRegressor()
    print("%-25s" % "DecisionTreeRegressor:", end="")
    test(model, features, x, y)

    model = RandomForestRegressor(n_estimators=200)
    print("%-25s" % "RandomForestRegressor:", end="")
    test(model, features, x, y)

    model = AdaBoostRegressor(n_estimators=50)
    print("%-25s" % "AdaBoostRegressor:", end="")
    test(model, features, x, y)

    model = neighbors.KNeighborsRegressor()
    print("%-25s" % "KNeighborsRegressor:", end="")
    test(model, features, x, y)
    '''
    '''
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
    '''

    '''
    sector_all = np.column_stack((sector2_1, sector3_5))
    X_train, X_test, Y_train, Y_test = train_test_split(sector_all, x_y, test_size=0.1, random_state=None)
    model1, model2 = split_train(X_train, list(Y_train['x']), list(Y_train['y']))
    predict_x, predict_y = split_predict(model1, model2, X_test)
    sum = 0.0
    for i in range(len(predict_x)):
        sum += math.sqrt((predict_x[i] - list(Y_test['x'])[i]) ** 2 + (predict_y[i] - list(Y_test['y'])[i]) ** 2)
        # print(predict_x[i], "\t", list(Y_test['x'])[i], "\t", predict_y[i], "\t", list(Y_test['y'])[i])
    print(sum / len(predict_x))

    '''
    test = pd.read_csv("./testAll.csv")
    sector_all = np.column_stack((sector2_1, sector3_5))
    model1, model2 = split_train(sector_all, list(x_y['x']), list(x_y['y']))
    predict_x, predict_y = split_predict(model1, model2, test.values)
    with open("./test_labels.csv", "w") as file:
        file.write("id,x,y\n")
        for i in range(len(predict_x)):
            file.write(str(i) + "," + str("%.3f" % predict_x[i]) + "," + str("%.3f" % predict_y[i]) + "\n")


if __name__ == '__main__':
    __main__()
