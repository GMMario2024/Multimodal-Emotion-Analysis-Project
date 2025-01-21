###############################################################################
# text.py 对文本进行特征提取，转换成向量
###############################################################################

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel

def convert_texts_without_feature(texts: list[str]) -> np.ndarray:
    '''
    将字符串**不经过特征提取**转换为向量。

    输入：
    - texts：要转换的字符串列表。
    输出：
    - 一个长为 N x l 的 numpy 数组，表示转换后的矩阵，其中 N 为字符串个数，
      l 为最长的字符串长度。
    '''
    max_length = max([len(s) for s in texts])
    mat = []
    for s in texts:
        vec = [ord(c) for c in s]
        vec.extend([0] * (max_length - len(s)))
        mat.append(vec)
    return np.array(mat)

def convert_tfidf(texts: list[str]) -> np.ndarray:
    '''
    将字符串通过 TF-IDF 的方式转换为向量。

    输入：
    - texts：要转换的字符串列表。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵，其中 N 为字符串个数，
      l 为向量长度。
    '''
    transformer = TfidfVectorizer()
    return transformer.fit_transform(texts).toarray()

def convert_word2vec(texts: list[str], vec_size: int=100, window: int=5, 
                     min_cnt: int=1) -> np.ndarray:
    '''
    将字符串通过 Word2Vec 的方式转换为向量。

    输入：
    - texts：要转换的字符串列表。
    - vec_size：转换的向量长度 l，默认为 100。
    - window：当前词与预测词的最大距离，默认为 5。
    - min_cnt：最小词频，默认为 1。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵，其中 N 为字符串个数，
      l 为向量长度。
    '''
    texts = [text.split() for text in texts]
    transformer = Word2Vec(texts, vector_size=vec_size, window=window, min_count=min_cnt)
    mat = []
    for text in texts:
        mat.append(np.mean([transformer.wv[word] for word in text], 0))
    return np.array(mat)

def convert_transformer(texts: list[str]):
    '''
    !!!实验功能：这种方式可能会变慢，且对内存要求高!!!
    通过 transformer 的 BERT 模型，将字符串转换成向量。

    输入：
    - texts：要转换的字符串列表。
    输出：
    - 一个 (N, l) 的 numpy 数组，表示转换后的矩阵，其中 N 为字符串个数，
      l 为向量长度。
    '''
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    transformer = AutoModel.from_pretrained('bert-base-uncased')
    mat = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = transformer(**inputs)
        last_hidden_states = outputs.last_hidden_state
        mat.append(last_hidden_states[0][0].detach().numpy())
    return np.array(mat)