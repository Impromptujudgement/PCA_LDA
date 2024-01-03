import os
from PIL import Image
import shutil
import numpy as np
import csv


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


def split_set(images, labels, num_people=40, images_per_person=10):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    for i in range(num_people):
        for j in range(images_per_person):
            index = i * images_per_person + j
            if j < images_per_person // 2:  # 取每个人前一半的图像作为训练集
                train_images.append(images[index])
                train_labels.append(labels[index])
            else:  # 取每个人后一半的图像作为测试集
                test_images.append(images[index])
                test_labels.append(labels[index])

    # 将列表转换为 NumPy 数组并归一化
    train_images = np.array(train_images, dtype='float32') / 255.0
    test_images = np.array(test_images, dtype='float32') / 255.0
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


def classify(test_sample, train_images, train_labels):
    distances = [euclidean_distance(test_sample, img) for img in train_images]
    min_distance_index = np.argmin(distances)
    return train_labels[min_distance_index]


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_predicted_labels(train_images, train_labels, test_images):
    # 这里假设train_images, train_labels, test_images已经被定义和初始化
    predicted_labels = []

    for img in test_images:
        predicted_label = classify(img, train_images, train_labels)
        predicted_labels.append(predicted_label)

    return predicted_labels


def visualise_results_to_csv(test_labels, predicted_labels, csv_filename):
    error_counts = [0] * len(np.unique(test_labels))

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # 写入表头
        csv_writer.writerow(["已知类号", "样本编号", "识别结果(类号)"])

        for i, (true_label, predicted_label) in enumerate(zip(test_labels, predicted_labels), start=1):
            error_status = "(error)" if true_label != predicted_label else ""
            # 写入数据行
            csv_writer.writerow([true_label, i, f"{predicted_label} {error_status}"])

            if true_label != predicted_label:
                error_counts[true_label - 1] += 1

    total_errors = sum(error_counts)
    error_rate = (total_errors / len(test_labels)) * 100

    # 输出每个类别的错误数、总的错误数和错误率
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            print("\t".join(row))
    for i, count in enumerate(error_counts, start=1):
        print(f"The number of errors of class {i} is {count}")
    print(f"The number of total errors is {total_errors}")
    print(f"The error rate is {error_rate:.2f} %")
    print_full_line("*")


def print_full_line(char='-'):
    terminal_width = shutil.get_terminal_size().columns
    line = char * terminal_width
    print(line)
