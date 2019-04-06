def textParse(bigString):  # input is big string, #output is word list
    """
        接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    """
        创建一个包含在所有文档中出现的不重复的词的列表。
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    """
        获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def createWordsBag():
    fileTexts = []
    for i in range(25):
        num = i + 1
        fileString = ""
        with open("spam/" + str(num) + ".txt", "r", encoding="iso-8859-1") as f:
            while (True):
                line = f.readline().replace("\r", "")
                if not line:
                    break
                fileString += line
            fileTexts.append(textParse(fileString))
        fileString = ""
        with open("ham/" + str(num) + ".txt", "r", encoding="iso-8859-1") as f:
            while (True):
                line = f.readline()
                if not line:
                    break
                fileString += line
            fileTexts.append(textParse(fileString))
    vocabList = createVocabList(fileTexts)
    fileVectors = []
    for file in fileTexts:
        fileVectors.append(bagOfWords2VecMN(vocabList, file))
    return fileVectors


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def main():
    trainVectors = createWordsBag()
    trainLabels = [0, 1] * 25
    nb = MultinomialNB()
    X_train, X_test, Y_train, Y_test = train_test_split(trainVectors, trainLabels, test_size=0.15, random_state=None)
    nb.fit(X_train, Y_train)
    print(nb.score(X_test, Y_test))


if __name__ == '__main__':
    main()
