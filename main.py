from LDA import *
from PCA_LDA import *
from MDC import *
from zero_LDA import *
from pca_reconstruct import *

if __name__ == '__main__':
    folder_path = './data-ORL/ORL/'  # 解压后数据所在路径
    images, labels = load_images(folder_path)
    train_images, train_labels, test_images, test_labels = split_set(images, labels)
    MDC(train_images, train_labels, test_images, test_labels)
    reconstruct(train_images)
    PCA_LDA(train_images, train_labels, test_images, test_labels)
    lda_pure(train_images, train_labels, test_images, test_labels)
    lda_pure(train_images, train_labels, test_images, test_labels, True)
    # zero_LDA(train_images, train_labels, test_images, test_labels)
