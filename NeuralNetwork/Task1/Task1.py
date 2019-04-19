from sklearn.neural_network import MLPClassifier

trainData = []
trainLabel = []
with open("./horseColicTraining.txt", 'r') as dataFile:
    while True:
        data = []
        line = dataFile.readline().replace("\n", "")
        if not line:
            break
        features = line.split()
        for i in range(len(features) - 1):
            data.append(float(features[i]))
        trainData.append(data)
        trainLabel.append(float(features[len(features) - 1]))

clf = MLPClassifier(solver='lbfgs', activation='logistic',
                    hidden_layer_sizes=(22, 6), random_state=1)
clf.fit(trainData, trainLabel)

testData = []
testLabel = []
with open("./horseColicTest.txt", 'r') as dataFile:
    while True:
        data = []
        line = dataFile.readline().replace("\n", "")
        if not line:
            break
        features = line.split()
        for i in range(len(features) - 1):
            data.append(float(features[i]))
        testData.append(data)
        testLabel.append(float(features[len(features) - 1]))

print(clf.score(testData, testLabel))
