from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime

# 贝叶斯网络作业 No.3
startTime = datetime.datetime.now()

# 读取训练集，使用one-hot编码
print("reading train_data.txt...")
trainDatas = []
with open("./train/train_data.txt") as trainDatasFile:
    while True:
        data = {}
        line = trainDatasFile.readline().replace("\n", "")
        if not line:
            break
        commentWords = line.split()
        for word in commentWords:
            if (int(word) - 1) in data:
                data[int(word) - 1] += 1
            else:
                data[int(word) - 1] = 1
        trainDatas.append(data)

trainDatas = pd.DataFrame(trainDatas).reindex(columns=range(10000)).fillna(0)

# 读取训练集标签
print("reading train_labels.txt...")
trainLabels = []
with open("./train/train_labels.txt") as trainLabelFile:
    while True:
        line = trainLabelFile.readline().replace("\n", "")
        if not line:
            break
        trainLabels.append(int(line))

print("word2vec...")
# 对训练集属性进行降维处理


# 获取需要预测的样本
testDatas = []
with open("./test/test_data.txt") as testDatasFile:
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

for i in range(10):
    clf = MultinomialNB(alpha=i * 1.0)
    clf = clf.fit(trainDatas, trainLabels)
    testLabels = clf.predict(testDatas)

    # 将预测值写入文本文件,共10次
    fileName = ""
    with open(fileName + str(i + 1) + ".txt", "w") as testLabelsFile:
        for label in testLabels:
            testLabelsFile.write(str(label))
            testLabelsFile.write("\n")
'''
# 留出法测试
for j in range(5):
    score = 0
    for i in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(trainDatas, trainLabels, test_size=0.2, random_state=None)
        clf = MultinomialNB(alpha=j * 0.5)
        clf.fit(X_train, Y_train)
        score += clf.score(X_test, Y_test)
    print("%.5f" % (score * 20.0))
'''
endTime = datetime.datetime.now()
print((endTime - startTime).seconds)  # 输出运行时间
