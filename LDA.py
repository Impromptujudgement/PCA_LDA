from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier

from util import *


def gram_schmidt(vectors):
    orthogonal = []
    for vector in vectors.T:
        for orth in orthogonal:
            vector -= np.dot(vector, orth) * orth
        vector /= np.linalg.norm(vector)
        orthogonal.append(vector)
    return np.array(orthogonal).T


def lda_pure(train_images, train_labels, test_images, test_labels, orthogonal=False):
    lda = LDA()  # 默认情况下，LDA 会使用最多 C-1 个组件，C 是类别数
    train_lda = lda.fit_transform(train_images, train_labels)
    test_lda = lda.transform(test_images)
    if orthogonal:
        discriminant_vectors = lda.scalings_
        orthogonal_discriminant_vectors = gram_schmidt(discriminant_vectors)
        # 把正交化后的鉴别向量应用到训练数据
        train_lda = np.dot(train_images - lda.xbar_, orthogonal_discriminant_vectors)
        test_lda = np.dot(test_images - lda.xbar_, orthogonal_discriminant_vectors)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_lda, train_labels)
    predicted_labels = classifier.predict(test_lda)
    visualise_results(test_labels, predicted_labels)
