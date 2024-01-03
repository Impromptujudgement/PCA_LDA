from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from util import *


def zero_lda(X, y, reg_param=0.01):
    lda = LDA(solver='eigen', shrinkage=reg_param)
    # lda = LDA(solver='svd', shrinkage=reg_param)
    return lda.fit_transform(X, y), lda


def zero_LDA(train_images, train_labels, test_images, test_labels):
    train_lda, lda_model = zero_lda(train_images, train_labels)

    # 将相同的 LDA 模型应用于测试数据
    test_lda = lda_model.transform(test_images)

    # 使用分类器进行分类
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_lda, train_labels)
    predicted_labels = classifier.predict(test_lda)
    visualise_results_to_csv(test_labels, predicted_labels, "./table/zero_LDA.csv")
