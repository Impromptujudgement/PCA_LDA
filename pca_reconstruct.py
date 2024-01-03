import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def apply_pca_and_reconstruct(images, n_components):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(images)
    reconstructed = pca.inverse_transform(transformed)
    return reconstructed, pca


def reconstruct(train_images):
    # 假设 train_images 是您的训练图像数据，已经被展平为一维向量
    # 假设每张图像的原始尺寸是 112x92
    height, width = 112, 92
    n_samples = len(train_images)
    train_images_reshaped = train_images.reshape(n_samples, -1)

    # 选择不同的 PCA 组件数以观察重构效果
    n_components_list = [10, 20, 30, 50, 100, 150, 200]
    errors = []

    for n_components in n_components_list:
        reconstructed_images, pca = apply_pca_and_reconstruct(train_images_reshaped, n_components)
        error = np.mean(np.square(train_images_reshaped - reconstructed_images))
        errors.append(error)

        # 显示第一张重构的图像作为示例
        example_image = reconstructed_images[0].reshape(height, width)
        plt.imshow(example_image, cmap='gray')
        plt.title(f"PCA with {n_components} components\nReconstruction Error: {error:.4f}")
        plt.savefig(f"fig/pca_reconstruction_{n_components}.png")
        plt.show()

    # 绘制误差图表
    plt.figure()
    plt.plot(n_components_list, errors, marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Reconstruction Error')
    plt.title('PCA Reconstruction Error vs. Number of Components')
    plt.savefig(f"fig/PCA_Reconstruction_Error")
    plt.show()
