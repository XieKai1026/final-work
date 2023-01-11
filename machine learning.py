# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from data_utils import load_CIFAR10
from sklearn.svm import SVC
import time
# ----------------------------------------------------------------------------------
# 第一步 切分训练集和测试集
# ----------------------------------------------------------------------------------

#加载Cifar10数据集，并输出数据集的维数
data_root = os.path.abspath(os.path.join(os.getcwd(), "cifar-10-python/cifar-10-batches-py"))
X_train,y_train,X_test,y_test = load_CIFAR10(data_root)
print('Training data shape', X_train.shape)
print('Training labels shape', y_train.shape)
print('Test data shape', X_test.shape)
print('Test labels shape', y_test.shape)
start = time.clock()
# ----------------------------------------------------------------------------------
# 第二步 图像读取及转换为像素直方图
# ----------------------------------------------------------------------------------
# 训练集
XX_train = np.reshape(X_train,(X_train.shape[0],-1))
XX_test = np.reshape(X_test,(X_test.shape[0],-1))
print(XX_train.shape,XX_test.shape)
# ----------------------------------------------------------------------------------
# 第三步 基于KNN的图像分类处理
# ----------------------------------------------------------------------------------
# clf = KNeighborsClassifier(n_neighbors=11).fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)


# ----------------------------------------------------------------------------------
# 第四步 基于支持向量机的图像分类处理
# ----------------------------------------------------------------------------------
#clf = SVC().fit(XX_train, y_train)
clf = SVC(kernel="rbf").fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test)
# ----------------------------------------------------------------------------------
# 第五步 基于决策树的图像分类处理
# ----------------------------------------------------------------------------------
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier().fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# 第六步 基于朴素贝叶斯的图像分类处理
# ----------------------------------------------------------------------------------
# 0.01
# from sklearn.naive_bayes import BernoulliNB
# clf = BernoulliNB().fit(XX_train, y_train)
# predictions_labels = clf.predict(XX_test)

print(u'预测结果:')
print(predictions_labels)
print(u'算法评价:')
print(classification_report(y_test, predictions_labels))
end = time.clock()
runTime = end - start
print("运行时间：", runTime)
