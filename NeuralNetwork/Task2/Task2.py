from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle

with open("./train/train_texts.dat", "rb") as file:
    train_texts = pickle.load(file)

# TF-IDF特征提取 + 转换为特征矩阵
vectorizer = TfidfVectorizer(max_features=10000)
train_datas = vectorizer.fit_transform(train_texts)

train_labels = []
with open("./train/train_labels.txt", "r") as file:
    while True:
        line = file.readline().replace("\n", "")
        if not line:
            break
        train_labels.append(int(line))

# train_datas = pd.DataFrame(train_datas)
# train_labels = pd.DataFrame(train_labels)

clf = MLPClassifier(solver='adam', activation='relu', verbose=True, early_stopping=True,
                    hidden_layer_sizes=(10000), random_state=1)
'''
# 留出法测试
X_train, X_test, Y_train, Y_test = train_test_split(train_datas, train_labels, test_size=0.2, random_state=None)
clf.fit(X_train, Y_train)
print("%.5f" % (clf.score(X_test, Y_test) * 100), "%")
'''

clf.fit(train_datas, train_labels)

with open("./test/test_texts.dat", "rb") as file:
    test_texts = pickle.load(file)

test_datas = vectorizer.transform(test_texts)

test_labels = clf.predict(test_datas)
for label in test_labels:
    print(test_labels)

file_name = "test_labels"
with open(file_name + ".txt", "w") as testLabelsFile:
    for label in test_labels:
        testLabelsFile.write(str(label))
        testLabelsFile.write("\n")
