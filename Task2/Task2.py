from sklearn import tree
import pandas as pd

data = pd.read_csv("TrainDT.csv", encoding="GB2312")  # 使用pandas加载训练
data = data.drop(columns=["SSIDLabel"])  # 去除无用的SSID列
trainDatas = []  # 用于存放处理过后的样本属性数据
trainFeatures = []  # 用于存放处理过后的样本标签
i = 0
while (True):
    BSSIDs = {}
    tmp = data.loc[data['finLabel'] == i + 1]  # 取出第(i+1)时刻被采集到的所有样本
    for line in tmp.values:
        BSSIDs[line[0]] = line[1]  # 将属性和取值转换为字典
    if len(BSSIDs) != 0:
        trainDatas.append(BSSIDs)  # 记录样本属性及取值
        trainFeatures.append(tmp.values[0][2])  # 记录样本标签
        i += 1
    else:
        break
trainDatas = pd.DataFrame(trainDatas).fillna(-100)  # 将列表转换为数据帧，并填充缺失值为-100
BSSIDLabels = trainDatas.columns.tolist()  # 取出列名

clf = tree.DecisionTreeClassifier(  # 定义决策树分类器
    criterion="gini",  # 使用基尼系数作为信息增益的计算标准
    splitter="best",  # 在特征的所有划分点中找出最优的划分点
    max_features="auto");  # 划分时最多考虑(√N)个特征
clf = clf.fit(trainDatas, trainFeatures)  # 生成模型

# 将决策树图形化
import graphviz

featureNames = ["room1", "room2", "room3", "room4"]
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=trainDatas.columns.tolist(),
                                class_names=featureNames,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("test", view=True)

testData = pd.read_csv("TestDT.csv", encoding="GB2312")  # 使用pandas加载测试集
testData = testData.drop(columns=["SSIDLabel"])  # 去除无用的SSID列
testDatas = []  # 用于存放处理过后的样本属性数据
testFeatures = []  # 用于存放处理过后的样本标签
i = 0
while (True):
    BSSIDs = {}
    tmp = testData.loc[testData['finLabel'] == i + 1]  # 取出第(i+1)时刻被采集到的所有样本
    for line in tmp.values:
        BSSID = line[0]
        # if BSSID in BSSIDLabels:
        BSSIDs[BSSID] = line[1]  # 将属性和取值转换为字典
    if len(BSSIDs) != 0:
        testDatas.append(BSSIDs)  # 记录样本属性及取值
        testFeatures.append(tmp.values[0][2])  # 记录样本标签
        i += 1
    else:
        break

testDatas = pd.DataFrame(testDatas)  # 转换为数据帧
testDatas = testDatas.reindex(columns=BSSIDLabels).fillna(-100)  # 重新建立列索引，保证与训练集相同，并填充缺失值为-100

predict = clf.predict(testDatas)  # 使用决策树预测样本
count = 0.0;
for i in range(len(testFeatures)):
    if testFeatures[i] == predict[i]:  # 比较预测结果和实际值
        count += 1
print(count / 109.0 * 100.0, "%")
