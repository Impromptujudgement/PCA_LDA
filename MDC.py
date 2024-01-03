# Nearest Neighbor Classifier or Minimum Distance Classifier
# Most Simple

from util import *


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def MDC(train_images, train_labels, test_images, test_labels):
    predicted_MDC = calculate_predicted_labels(train_images, train_labels, test_images)
    visualise_results(test_labels, predicted_MDC)
