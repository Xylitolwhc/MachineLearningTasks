from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import datetime

# 决策树作业 No.3
startTime = datetime.datetime.now()

#
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

clf = tree.DecisionTreeClassifier(  # 定义决策树分类器
    criterion="gini",  # 使用基尼系数作为信息增益的计算标准
    splitter="best",  # 在特征的所有划分点中找出最优的划分点
    min_impurity_decrease=0.0004,  # 设置节点信息增益阈值，当小于此值时即停止划分节点
    max_features=None);  # 划分时考虑所有特征，计算量大，耗时更长
clf = clf.fit(trainDatas, trainLabes)

'''
# 获取需要预测的样本
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
'''

X_train, X_test, Y_train, Y_test = train_test_split(trainDatas, trainLabes, test_size=0.5, random_state=None)
print(svm.SVC(kernel='linear').fit(X_train, Y_train).score(X_test, Y_test))

'''
clf = svm.SVC(kernel='linear').fit(trainDatas, trainLabes)
testlabels = clf.predict(testDatas)
#将预测值写入文本文件
with open("test_labels.txt","w") as testLabelsFile:
    for label in testlabels:
        testLabelsFile.write(str(label))
        testLabelsFile.write("\n")
'''

endTime = datetime.datetime.now()
print((endTime - startTime).seconds)  # 输出运行时间
