from sklearn import tree
import pandas as pd
import graphviz

# 决策树 No.2
# 读取并处理训练集
data = pd.read_csv("TrainDT.csv", encoding="GB2312")  # 使用pandas加载训练集文件
data = data.drop(columns=["SSIDLabel"])  # 去除无用的SSID列
trainDatas = []  # 用于存放处理过后的样本属性数据
trainLabels = []  # 用于存放处理过后的样本标签
i = 0
while (True):
    BSSIDs = {}
    tmp = data.loc[data['finLabel'] == i + 1]  # 取出第(i+1)时刻被采集到的所有样本
    for line in tmp.values:
        BSSIDs[line[0]] = line[1]  # 将属性和取值转换为字典
    if len(BSSIDs) != 0:
        trainDatas.append(BSSIDs)  # 记录样本属性及取值
        trainLabels.append(tmp.values[0][2])  # 记录样本标签
        i += 1
    else:
        break

trainDatas = pd.DataFrame(trainDatas).fillna(-100)  # 将列表转换为数据帧，并填充缺失值为-100

# 训练模型
clf = tree.DecisionTreeClassifier(  # 定义决策树分类器
    criterion="gini",  # 使用基尼系数作为信息增益的计算标准
    splitter="best",  # 在特征的所有划分点中找出最优的划分点
    max_features="auto");  # 划分时最多考虑(√N)个特征
clf = clf.fit(trainDatas, trainLabels)  # 生成模型

# 将决策树图形化
featureNames = ["room1", "room2", "room3", "room4"]
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=trainDatas.columns.tolist(),
                                class_names=featureNames,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree", view=True)

# 准备测试集
testData = pd.read_csv("TestDT.csv", encoding="GB2312")  # 使用pandas加载测试集
testData = testData.drop(columns=["SSIDLabel"])  # 去除无用的SSID列
testDatas = []  # 用于存放处理过后的样本属性数据
testLabels = []  # 用于存放处理过后的样本标签
i = 0
while (True):
    BSSIDs = {}
    tmp = testData.loc[testData['finLabel'] == i + 1]  # 取出第(i+1)时刻被采集到的所有样本
    for line in tmp.values:
        BSSID = line[0]
        BSSIDs[BSSID] = line[1]  # 将属性和取值转换为字典
    if len(BSSIDs) != 0:
        testDatas.append(BSSIDs)  # 记录样本属性及取值
        testLabels.append(tmp.values[0][2])  # 记录样本标签
        i += 1
    else:
        break

testDatas = pd.DataFrame(testDatas)  # 转换为数据帧
BSSIDLabels = trainDatas.columns.tolist()  # 取出训练集列名
testDatas = testDatas.reindex(columns=BSSIDLabels).fillna(-100)  # 重新建立列索引，保证与训练集相同，并填充缺失值为-100

# 输出模型预测精度
print(clf.score(testDatas, testLabels) * 100, "%")
