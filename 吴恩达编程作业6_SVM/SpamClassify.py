# SVM --- Spam Classification 垃圾邮件分类
# Feb.28th
# 有两个步骤没有做：1.邮件的预处理；
#               2.对处理后的邮件提取特征（根据vocab.txt文件）
# 处理完上述两个步骤后，就是一般的SVM问题了

import scipy.io as scio
from sklearn import svm

# 导入数据
email_train = scio.loadmat('spamTrain.mat')['X']
label_train = scio.loadmat('spamTrain.mat')['y'].reshape(-1)
email_test = scio.loadmat('spamTest.mat')['Xtest']
label_test = scio.loadmat('spamTest.mat')['ytest'].reshape(-1)

# 训练
clf = svm.SVC(kernel='rbf', C=1, gamma=50)
clf.fit(email_train, label_train)

# 计算得分
score_train = clf.score(email_train, label_train)
score_test = clf.score(email_test, label_test)
print('训练集得分：', score_train)
print('测试集得分：', score_test)

# 读取vocab.txt文件
vocab = {'index': [], 'word': []}
f = open('vocab.txt', 'r')
for line in f:
    for i in range(len(line)):
        if line[i] == '\t':
            vocab['index'].append(int(line[:i]))
            vocab['word'].append(line[i+1:len(line)-1])
            break






