# Nearest Neighbor Classifier or Minimum Distance Classifier
# Most Simple

import numpy as np
from util import *


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

#
# def calculate_predicted_labels(train_images, train_labels, test_images):
#     # 这里假设train_images, train_labels, test_images已经被定义和初始化
#     predicted_labels = []
#
#     for img in test_images:
#         predicted_label = classify(img, train_images, train_labels)
#         predicted_labels.append(predicted_label)
#
#     return predicted_labels
#
#
# def classify(test_sample, train_images, train_labels):
#     distances = [euclidean_distance(test_sample, img) for img in train_images]
#     min_distance_index = np.argmin(distances)
#     return train_labels[min_distance_index]


def MDC(train_images, train_labels, test_images, test_labels):
    predicted_MDC = calculate_predicted_labels(train_images, train_labels, test_images)
    visualise_results(test_labels, predicted_MDC)
