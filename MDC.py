# Nearest Neighbor Classifier or Minimum Distance Classifier
# Most Simple

from util import *


def MDC(train_images, train_labels, test_images, test_labels):
    predicted_MDC = calculate_predicted_labels(train_images, train_labels, test_images)
    visualise_results_to_csv(test_labels, predicted_MDC, './table/MDC.csv')
