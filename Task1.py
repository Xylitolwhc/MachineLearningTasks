from math import log


class Tree:

    def __init__(self):
        self.endNode = False
        self.decisionFeature = -1;
        self.featureDecision = {}
        self.decision = ""

    def createTree(self, datas, features):
        if len(features) == 0:
            self.endNode = True;
            self.decision += datas[0][-1]
            return self
        maxGain = 0.0
        selectedFeature = -1
        selectedDatas = []
        selectedFeatures = {}
        newDatas = []
        featureValues = {}
        # 遍历查找最优划分属性
        for feature in features:
            [gain, newDatas, featureValues] = self.calGain(datas, feature)
            if gain > maxGain:
                maxGain = gain
                selectedDatas = newDatas
                selectedFeatures = featureValues
                selectedFeature = feature

        if maxGain <= 0.0:
            self.endNode = True;
            self.decision = datas[0][-1]
            return self

        # 移除已划分属性
        leftFetures = list(features)
        leftFetures.remove(selectedFeature)

        # tree = "\n("
        # for featureValue in selectedFeatures:
        #     tree += str(featureValue) + "/"
        # tree = tree[:-1] + "):"
        # for i in range(len(selectedDatas)):
        #     tree += "\n\t"
        #     tree += list(selectedFeatures.keys())[i]
        #     tree += " -> "
        #     tree += self.createTree(selectedDatas[i], leftFetures).replace("\n", "\n\t")  # 递归划分属性
        # return tree
        for i in range(len(selectedDatas)):
            self.decisionFeature = selectedFeature
            self.endNode = False
            tmp = Tree()
            self.featureDecision[list(selectedFeatures.keys())[i]] = tmp.createTree(selectedDatas[i], leftFetures)
        return self

    # 计算信息增益，并返回相应属性划分后的样本
    def calGain(self, datas, feature):
        size = len(datas)
        ent = self.calEnt(datas)
        if size == 0 or size == 1:
            return [0, datas, []]
        entv = 0
        dividedDatas = []
        # 找出属性的所有取值
        featureCount = self.calFeatureCount(datas, feature)
        if len(list(featureCount.keys())) == 1:
            return [0, datas, featureCount]
        # 遍历属性的所有取值，计算该属性的信息增益
        for value in featureCount.keys():
            featuredDatas = []
            for data in datas:
                if data[feature] == value:
                    featuredDatas.append(data)
            dividedDatas.append(featuredDatas)
            # 计算Ent(Dv)
            entv += self.calEnt(featuredDatas) * featureCount[value] / size
        return [ent - entv, dividedDatas, featureCount]

    # 计算信息熵
    def calEnt(self, datas):
        ent = 0.0
        size = len(datas)
        valueCount = self.calFeatureCount(datas, len(datas[0]) - 1)
        # 计算熵
        for key in valueCount.keys():
            # print(key, ":", valueCount[key])
            num = valueCount[key]
            ent -= (num / size) * log(num / size, 2)
        return ent

    # 用于统计某个属性划分后各取值的个数
    def calFeatureCount(self, datas, feature):
        valueCount = {}
        for data in datas:
            value = data[feature]
            if value in valueCount.keys():
                valueCount[value] += 1
            else:
                valueCount[value] = 1
        return valueCount


def __main__():
    # 读取数据文件并转换为列表
    trainData = []
    with open("lenses.txt", 'r') as dataFile:
        while True:
            line = dataFile.readline().replace("\n", "")
            if not line:
                break
            trainData.append(line.split("\t"))
    features = range(len(trainData[0]) - 1)
    root = Tree()
    root = root.createTree(trainData, features)
    print(printTree(root))


def printTree(node):
    tree = ""
    if node.endNode:
        tree += node.decision
    else:
        tree = "\n" + str(node.decisionFeature) + ":("
        for key in list(node.featureDecision.keys()):
            tree += key + "/"
        tree = tree[:-1] + ")"
        for key in list(node.featureDecision.keys()):
            tree += "\n\t" + key + " -> "
            tree += printTree(node.featureDecision[key]).replace("\n", "\n\t")
    return tree


if __name__ == '__main__':
    __main__()
