###############################################################################
# image.py 对图片进行特征提取，转换成向量
###############################################################################

import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv
import torch
from torchvision.models import alexnet
import torchvision.transforms as T

def convert_images_without_feature(imgs: np.ndarray) -> np.ndarray:
    '''
    将图片**不经过特征提取**转换为向量。

    输入：
    - imgs：要转换的 numpy 矩阵，大小为 (N, 224, 224, 3)，其中 N 为样本数。
    输出：
    - 一个 (N, 224 * 224 * 3) 的 numpy 数组，表示转换后的矩阵。
    '''
    return imgs.reshape((imgs.shape[0], 224 * 224 * 3))

def convert_hog(imgs: np.ndarray) -> np.ndarray:
    '''
    将图片通过 HOG 的方式提取特征，转换为向量。

    输入：
    - imgs：要转换的 numpy 矩阵，大小为 (N, 224, 224, 3)，其中 N 为样本数。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵。
    '''
    mat = [hog(rgb2gray(imgs[i])) for i in range(imgs.shape[0])]
    return np.array(mat)

def convert_hsv(imgs: np.ndarray) -> np.ndarray:
    '''
    将图片通过 HSV 的方式提取特征，转换为向量。

    输入：
    - imgs：要转换的 numpy 矩阵，大小为 (N, 224, 224, 3)，其中 N 为样本数。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵。
    '''
    mat = [rgb2hsv(imgs[i]).flatten() for i in range(imgs.shape[0])]
    return np.array(mat)

def convert_alexnet(imgs: np.ndarray) -> np.ndarray:
    '''
    !!!实验功能：这种方式可能会变慢，且对内存要求高!!!
    将图片通过 AlexNet 的方式提取特征，转换为向量。

    输入：
    - imgs：要转换的 numpy 矩阵，大小为 (N, 224, 224, 3)，其中 N 为样本数。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵。
    '''
    transformer = alexnet(pretrained=True)
    transformer.eval()
    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
    imgs = torch.stack([transform(img) for img in imgs])
    with torch.no_grad():
        mat = transformer.feature(imgs)
        mat = transformer.avgpool(mat)
        mat = torch.flatten(mat, 1)
    return mat.cpu().numpy()