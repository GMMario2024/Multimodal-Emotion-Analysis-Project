###############################################################################
# hybrid.py 对多模态特征进行聚合
###############################################################################

import numpy as np

def concatenate(mat_1: np.ndarray, mat_2: np.ndarray) -> np.ndarray:
    '''
    直接拼接两个矩阵实现特征聚合。

    输入：
    - mat_1：一个 (N, a) 的 numpy 数组。
    - mat_2：一个 (N, b) 的 numpy 数组。
    输出：
    - 一个 (N, a + b) 的 numpy 数组。
    '''
    if not mat_1.ndim == 2:
        raise ValueError(f'mat_1.ndim = {mat_1.ndim}: should be 2!')
    if not mat_2.ndim == 2:
        raise ValueError(f'mat_2.ndim = {mat_2.ndim}: should be 2!')
    if not mat_1.shape[0] == mat_2.shape[0]:
        raise ValueError(f'col_1 {mat_1.shape[0]} != col_2 {mat_2.shape[0]}!')
    return np.concatenate((mat_1, mat_2), 1)