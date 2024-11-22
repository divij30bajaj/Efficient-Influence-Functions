import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def plot(layer_num):
    activations = torch.load('layer_{}.pt'.format(layer_num))
    start = 0
    x = activations["ab_stereo"][start:]
    x = [tensor.detach().to(torch.float).numpy() for tensor in x]
    x1 = np.concatenate(x, axis=0)

    x = activations["c_neutral"][start:]
    x = [tensor.detach().to(torch.float).numpy() for tensor in x]
    x2 = np.concatenate(x, axis=0)

    x = activations["ab_neutral"][start:]
    x = [tensor.detach().to(torch.float).numpy() for tensor in x]
    x3 = np.concatenate(x, axis=0)

    x = activations["c_stereo"][start:]
    x = [tensor.detach().to(torch.float).numpy() for tensor in x]
    x4 = np.concatenate(x, axis=0)

    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)
    mean3 = np.mean(x3, axis=0)
    mean4 = np.mean(x4, axis=0)

    dir1 = mean1 - mean3
    dir2 = mean2 - mean4

    print(cosine_similarity(dir1.reshape(1, -1), dir2.reshape(1, -1))[0,0])
    length = x1.shape[0]

    pca = PCA(n_components=2)
    x = np.concatenate((x1, x2, x3, x4), axis=0)
    reduced_data = pca.fit_transform(x)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:length, 0], reduced_data[:length, 1], color='blue', marker='o')
    plt.scatter(reduced_data[length:length*2, 0], reduced_data[length:length*2, 1], color='blue', marker='x')
    plt.scatter(reduced_data[length*2:length*3, 0], reduced_data[length*2:length*3, 1], color='red', marker='o')
    plt.scatter(reduced_data[length*3:, 0], reduced_data[length*3:, 1], color='red', marker='x')
    plt.title('Layer {}'.format(layer_num))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Template 1', 'Template 2', 'Template 3', 'Template 4'])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot(5) # 2 or 5