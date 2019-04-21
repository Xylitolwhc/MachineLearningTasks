import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import copy


def accuracy(original_x, predict_x, original_y, predict_y):
    if len(original_y) != len(predict_y) \
            or len(original_x) != len(predict_x) \
            or len(original_x) != len(original_y):
        return
    return np.average(np.sqrt(np.add(
        np.power(np.subtract(original_y, predict_y), 2),
        np.power(np.subtract(original_x, predict_x), 2)
    )))


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
    columns = []
    for i in range(15):
        columns.append("2100" + str(i))
    sector2_1 = data[columns]

    for i in range(15):
        columns.append("3500" + str(i))
    sector3_5 = data[columns]

    features = sector3_5

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


if __name__ == '__main__':
    __main__()
