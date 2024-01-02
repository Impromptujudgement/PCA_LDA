import os
from PIL import Image
import shutil
import numpy as np


def load_images(folder_path):
    images = []
    labels = []
    for i in range(1, 41):  # 对于每个人
        for j in range(1, 11):  # 对于每个人的每张照片
            image_path = os.path.join(folder_path, f'orl{(i - 1) * 10 + j:03d}.bmp')
            if os.path.isfile(image_path):  # 确保文件存在
                image = Image.open(image_path)
                images.append(np.array(image).flatten())  # 将图像转换为一维数组
                labels.append(i)  # 标签为人的编号
    return np.array(images), np.array(labels)


def split_set(images, labels):
    train_images = images[::2]  # 奇数编号图像作为训练集
    train_images = train_images.astype('float32')
    train_images /= 255.0
    train_labels = labels[::2]

    test_images = images[1::2]  # 偶数编号图像作为测试集
    test_images = test_images.astype('float32')
    test_images /= 255.0
    test_labels = labels[1::2]
    return train_images, train_labels, test_images, test_labels


def calculate_predicted_labels(train_images, train_labels, test_images):
    # 这里假设train_images, train_labels, test_images已经被定义和初始化
    predicted_labels = []

    for img in test_images:
        predicted_label = classify(img, train_images, train_labels)
        predicted_labels.append(predicted_label)

    return predicted_labels


def visualise_results(test_labels, predicted_labels):
    error_counts = [0] * len(np.unique(test_labels))
    # 输出已知类号、样本编号和识别结果
    print("已知类号\t样本编号\t识别结果(类号)")
    for i, (true_label, predicted_label) in enumerate(zip(test_labels, predicted_labels), start=1):
        error_status = "(error)" if true_label != predicted_label else ""
        print(f"{true_label}\t{i}\t{predicted_label} {error_status}")
        if true_label != predicted_label:
            error_counts[true_label - 1] += 1
    # 计算总的错误数和错误率
    total_errors = sum(error_counts)
    error_rate = (total_errors / len(test_labels)) * 100

    # 输出每个类别的错误数、总的错误数和错误率
    for i, count in enumerate(error_counts, start=1):
        print(f"The number of errors of class {i} is {count}")
    print(f"The number of total errors is {total_errors}")
    print(f"The error rate is {error_rate:.2f} %")
    print_full_line('*')


def print_full_line(char='-'):
    terminal_width = shutil.get_terminal_size().columns
    line = char * terminal_width
    print(line)


def classify(test_sample, train_images, train_labels):
    distances = [euclidean_distance(test_sample, img) for img in train_images]
    min_distance_index = np.argmin(distances)
    return train_labels[min_distance_index]


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
