###############################################################################
# classifier.py：建立分类模型
###############################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_knn_classifier(k: int=5, p: int=2):
    '''
    获取 kNN 分类器。

    输入：
    - k：邻居的个数，默认为 5。
    - p：计算距离采用的方式，默认为 2。
    输出：
    - 一个 kNN 分类器。
    '''
    if k <= 0:
        raise ValueError(f'k = {k}: should be positive integer!')
    if p <= 0:
        raise ValueError(f'p = {p}: should be positive integer!')
    classifier = KNeighborsClassifier(k, p=p)
    return classifier

def get_bayes_classifier():
    '''
    获取朴素贝叶斯分类器。

    输入：无。
    输出：
    - 一个朴素贝叶斯分类器。
    '''
    classifier = GaussianNB()
    return classifier

def get_decision_tree_classifier():
    '''
    获取决策树分类器。

    输入：无。
    输出：
    - 一个决策树分类器。
    '''
    classifier = DecisionTreeClassifier()
    return classifier

def get_svm_classifier():
    '''
    获取 SVM 分类器。

    输入：无。
    输出：
    - 一个 SVM 分类器。
    '''
    classifier = SVC()
    return classifier

def get_mlp_classifier(hidden_dim: tuple[int]=(100, ), activiation: str='relu',
                       optim: str='adam', learning_rate: int=0.001, 
                       early_stop: bool=False):
    '''
    获取 MLP 分类器。

    输入：
    - hidden_dim：隐藏神经元个数，默认为 (100, )。
    - activiation：激活函数，默认为 'relu'。
    - optim：优化器，默认为 'adam'。
    - learning_rate：学习率，默认为 0.001。
    - early_stop：是否早停，默认为 False。
    输出：
    - 一个 MLP 分类器。
    '''
    classifier = MLPClassifier(hidden_dim, activiation, solver=optim, 
                               learning_rate_init=learning_rate, 
                               early_stopping=early_stop)
    return classifier