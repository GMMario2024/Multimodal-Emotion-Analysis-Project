###############################################################################
# data.py：处理文本数据、图像数据以及训练数据的输入
###############################################################################

import os
from PIL import Image
import numpy as np 

def read_list(path: str, num: int=1e9) -> (list[str], list[str]):
    '''
    读取包含训练或测试的文件名和标签信息的文本文件。

    输入：
    - path：存储文本文件的路径。
    - num：存储样本数，默认为 1e9（无穷大）。
    输出：
    - files：存储所有文件名字符串组成的列表。
    - tags：存储对应的标签字符串组成的列表。
    '''
    if num <= 0:
        raise ValueError(f'num = {num}: should be positive integer!')
    if not path.endswith('txt'):
        raise ValueError(f'{path}: should end with .txt!')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path}: file not found!')
    files = []
    tags = []
    with open(path, 'r', encoding='utf-8') as file:
        if not file.readline().strip() == 'guid,tag':
            raise ValueError(f'{path}: format in file is wrong!')
        cnt = 0 
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 2:
                raise ValueError(f'{path}: format in file is wrong!')
            files.append(parts[0])
            tags.append(parts[1])
            cnt += 1
            if cnt == num: 
                break
    return files, tags

def read_txt(path: str) -> str:
    '''
    读取文本文件的数据，并转化为字符串。

    输入：
    - path：存储字符串的路径（需要包含文件的 .txt 扩展名），注意如果不是完整路径，
      则系统会以执行的 python 文件的目录为根目录。
    输出：
    - 一个字符串，存储文本文件的所有信息。
    '''
    if not path.endswith('txt'):
        raise ValueError(f'{path}: should end with .txt!')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path}: file not found!')
    with open(path, 'r', encoding='GB18030') as file:
        return file.read()

def read_jpg(path: str) -> np.ndarray:
    '''
    读取图像文件的数据，并转换为 numpy 数组。

    输入：
    - path：存储字符串的路径（需要包含文件的 .jpg 扩展名），注意如果不是完整路径，
      则系统会以执行的 python 文件的目录为根目录。
    输出：
    - 一个 (224, 224, 3) 的 numpy 数组，注意图像会被调整为 224 x 224。
    '''
    if not path.endswith('jpg'):
        raise ValueError(f'{path}: should end with .jpg!')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path}: file not found!')
    img = Image.open(path).resize((224, 224)).convert('RGB')
    return np.array(img)

def read_texts(root: str, files: list[str]) -> list[str]:
    '''
    按照输入的文件名列表读取 data 文件夹下的 .txt 文件，并转化为字符串列表。

    输入：
    - root：存储文件的文件夹。注意如果不是完整路径，则系统会以执行的 python 文
      件的目录为根目录。
    - files：文件名列表（文件名不包含 .txt 扩展名）。
    输出：
    - 一个字符串列表，存储 .txt 文件的文本数据。
    '''
    if not os.path.exists(root):
        raise ValueError(f'{root}: directory not found!')
    if not os.path.isdir(root):
        raise ValueError(f'{root}: should be directory!')
    texts = []
    for file in files:
        path = root + '/' + file + '.txt'
        texts.append(read_txt(path))
    return texts

def read_imgs(root: str, files: list[str]) -> np.ndarray:
    '''
    按照输入的文件名列表读取 data 文件夹下的 .jpg 文件，并转化为字符串列表。

    输入：
    - root：存储文件的文件夹。注意如果不是完整路径，则系统会以执行的 python 文
      件的目录为根目录。
    - files：文件名列表（文件名不包含 .jpg 扩展名）。
    输出：
    - 一个 (N, 224, 224, 3) 的 numpy 数组，存储所有图片的原始数据。
    '''
    if not os.path.exists(root):
        raise ValueError(f'{root}: directory not found!')
    if not os.path.isdir(root):
        raise ValueError(f'{root}: should be directory!')
    imgs = []
    for file in files:
        path = root + '/' + file + '.jpg'
        imgs.append(read_jpg(path))
    return np.stack(imgs, 0)