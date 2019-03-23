from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
from sklearn import svm
import datetime

startTime = datetime.datetime.now()

trainDatas = []
with open("train_data.txt") as trainDatasFile:
    while True:
        data = {}
        line = trainDatasFile.readline().replace("\n", "")
        if not line:
            break
        commentWords = line.split()
        for word in commentWords:
            data[int(word) - 1] = 1
        trainDatas.append(data)
trainDatas = pd.DataFrame(trainDatas).reindex(columns=range(10000)).fillna(0)

trainLabes = []
with open("train_labels.txt") as trainLabelFile:
    while True:
        line = trainLabelFile.readline().replace("\n", "")
        if not line:
            break
        trainLabes.append(int(line))
'''
X_train, X_test, Y_train, Y_test = train_test_split(trainDatas, trainLabes, test_size=0.1, random_state=None)
for j in range(20):
    minImpurityDecrease = 0.00005 * j
    clf = tree.DecisionTreeClassifier(  # 定义决策树分类器
        criterion="gini",  # 使用基尼系数作为信息增益的计算标准
        splitter="best",  # 在特征的所有划分点中找出最优的划分点
        min_impurity_decrease=minImpurityDecrease,  # 设置节点信息增益阈值，当小于此值时即停止划分节点
        max_features=None);  # 划分时最多考虑(√N)个特征
    clf.fit(X_train, Y_train)
    print("%.5f" % minImpurityDecrease, ":", "%.5f" % clf.score(X_test,Y_test))
'''

clf = tree.DecisionTreeClassifier(  # 定义决策树分类器
    criterion="gini",  # 使用基尼系数作为信息增益的计算标准
    splitter="best",  # 在特征的所有划分点中找出最优的划分点
    min_impurity_decrease=0.0004,  # 设置节点信息增益阈值，当小于此值时即停止划分节点
    max_features=None);  # 划分时最多考虑(√N)个特征
clf = clf.fit(trainDatas, trainLabes)

# clf = svm.SVC(kernel='linear').fit(X_train, Y_train)
# print(clf.score(X_test, Y_test))

'''
# 将决策树图形化
featureNames = ["Negative", "Positive"]
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=trainDatas.columns.tolist(),
                                class_names=featureNames,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree", view=True)
'''

testDatas = []
with open("test_data.txt") as testDatasFile:
    while True:
        data = {}
        line = testDatasFile.readline().replace("\n", "")
        if not line:
            break
        commentWords = line.split()
        for word in commentWords:
            data[int(word) - 1] = 1
        testDatas.append(data)
testDatas = pd.DataFrame(testDatas).reindex(columns=range(10000)).fillna(0)
X_train, X_test, Y_train, Y_test = train_test_split(trainDatas, trainLabes, test_size=0.5, random_state=None)
print(svm.SVC(kernel='linear').fit(X_train, Y_train).score(X_test, Y_test))
'''
clf = svm.SVC(kernel='linear').fit(trainDatas, trainLabes)
testlabels = clf.predict(testDatas)


with open("test_labels.txt","w") as testLabelsFile:
    for label in testlabels:
        testLabelsFile.write(str(label))
        testLabelsFile.write("\n")
'''

endTime = datetime.datetime.now()
print((endTime - startTime).seconds)
