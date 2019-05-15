import pandas as pd
from sklearn.cluster import DBSCAN,KMeans

# 读取并处理训练集
data = pd.read_csv("./DataSetKMeans2.csv", encoding="GB2312")  # 使用pandas加载训练集文件
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

db = KMeans(n_clusters=4).fit(trainDatas)
labels = db.labels_
for i in range(len(trainLabels)):
    print(str(labels[i]) + "\t" + str(trainLabels[i]))
