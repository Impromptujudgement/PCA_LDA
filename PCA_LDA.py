# PCA+LDA


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from util import *


def PCA_LDA(train_images, train_labels, test_images, test_labels):
    # 这里假设train_images, train_labels, test_images, test_labels已经被定义和初始化
    pca = PCA(n_components=115)  # max N-C 160
    train_images_pca = pca.fit_transform(train_images)
    test_images_pca = pca.transform(test_images)

    # 应用 LDA
    lda = LDA(n_components=39)  # max C-1 39
    train_images_lda = lda.fit_transform(train_images_pca, train_labels)
    test_images_lda = lda.transform(test_images_pca)
    predicated_PCA_LDA = calculate_predicted_labels(train_images_lda, train_labels, test_images_lda)
    visualise_results_to_csv(test_labels, predicated_PCA_LDA, "./table/PCA_LDA.csv")
