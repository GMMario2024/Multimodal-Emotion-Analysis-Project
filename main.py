###############################################################################
# main.py 程序运行的入口
###############################################################################

from utils.data import read_list, read_texts, read_imgs
from utils.feature.text import *
from utils.feature.image import *
from utils.feature.hybrid import concatenate
from utils.classifier import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('##################################')
print('# 多模态情感分析程序')
print('##################################')

# 第一步：读取 train.txt 和 test_without_label.txt 中的文件和标签信息。
print('正在读取 in/train.txt...')
train_files, y = read_list('in/train.txt')
print('正在读取 input/test_without_label.txt...')
test_files, _ = read_list('input/test_without_label.txt')
print('文件名与标签数据读取完成。')

# 第二步：读取文本。
print('正在读取训练文本...')
train_texts = read_texts('in/data', train_files)
print('正在读取测试文本...')
test_texts = read_texts('in/data', test_files)
texts = train_texts + test_texts
print('文本数据读取完成。')

# 第三步：将文本进行特征转换。
print('请输入数字以选择文本特征提取方式。')
print('- 1：不进行特征提取，直接转换。')
print('- 2：用 TF-IDF 进行特征提取。')
print('- 3：用 Word2Vec 进行特征提取（推荐）。')
print('- 4（实验）：用 transformers 的 BERT 模型提取。')
flag = int(input())
if flag != 1 and flag != 2 and flag != 3 and flag != 4:
    raise ValueError(f'input {flag}: should be 1, 2, 3 or 4!')
else:
    print('正在转换文本特征...')
    if flag == 1:
        X_text = convert_texts_without_feature(texts)
    elif flag == 2:
        X_text = convert_tfidf(texts)
    elif flag == 3:
        X_text = convert_word2vec(texts)
    else:
        X_text = convert_transformer(texts)
X_train_text = X_text[:len(train_texts)]
X_test_text = X_text[len(train_texts):]
print('文本数据已成功转换为矩阵。')

# 第四步：读取图像，并将图像进行特征转换。
print('正在读取训练图像...')
train_imgs = read_imgs('input/data', train_files)
print('正在读取测试图像...')
test_imgs = read_imgs('input/data', test_files)
print('图像数据读取完成。')
print('请输入数字以选择图像特征提取方式。')
print('- 1：不进行特征提取，直接转换。')
print('- 2：用 HOG 进行特征提取（推荐）。')
print('- 3：用 HSV 进行特征提取。')
print('- 4（实验）：用 AlexNet 模型提取。')
flag = int(input())
if flag != 1 and flag != 2 and flag != 3 and flag != 4:
    raise ValueError(f'input {flag}: should be 1, 2, 3 or 4!')
else:
    print('正在转换图像特征...')
    if flag == 1:
        X_train_img = convert_images_without_feature(train_imgs)
        X_test_img = convert_images_without_feature(test_imgs)
    elif flag == 2:
        X_train_img = convert_hog(train_imgs)
        X_test_img = convert_hog(test_imgs)
    elif flag == 3:
        X_train_img = convert_hsv(train_imgs)
        X_test_img = convert_hsv(test_imgs)
    else:
        X_train_img = convert_alexnet(train_imgs)
        X_test_img = convert_alexnet(test_imgs)
print('图像数据已成功转换为矩阵。')

# 第五步：整合特征，将数据进行拼接。
print('正在聚合特征...')
X_train = concatenate(X_train_img, X_train_text)
X_test = concatenate(X_test_img, X_test_text)
print('特征聚合完成。')

# 第六步：划分验证集。
print('正在划分验证集...')
X_train, X_val, y_train, y_val = train_test_split(X_train, y, random_state=1)
X_train_text, X_val_text, _, _ = train_test_split(X_train_text, y, random_state=1)
X_train_img, X_val_img, _, _ = train_test_split(X_train_img, y, random_state=1)
print('验证集划分完成。')

# 第七步：选择一个模型进行训练。
print('请输入数字以选择分类器。')
print('- 1：kNN 分类器（推荐）。')
print('- 2：朴素贝叶斯分类器。')
print('- 3：决策树分类器。')
print('- 4：SVM 分类器。')
print('- 5（实验）：MLP 分类器。')
flag = int(input())
if flag != 1 and flag != 2 and flag != 3 and flag != 4 and flag != 5:
    raise ValueError(f'input {flag}: should be 1, 2, 3, 4 or 5!')
else:
    if flag == 1:
        classifier = get_knn_classifier()
    elif flag == 2:
        classifier = get_bayes_classifier()
    elif flag == 3:
        classifier = get_decision_tree_classifier()
    elif flag == 4:
        classifier = get_svm_classifier()        
    else:
        classifier = get_mlp_classifier
classifier.fit(X_train, y_train)
y_pred_val = classifier.predict(X_val)
acc_score = accuracy_score(y_val, y_pred_val)
print('所用模型在聚合矩阵下的准确率：', acc_score)
classifier.fit(X_train_text, y_train)
y_pred_val = classifier.predict(X_val_text)
acc_score = accuracy_score(y_val, y_pred_val)
print('所用模型在文本矩阵下的准确率：', acc_score)
classifier.fit(X_train_img, y_train)
y_pred_val = classifier.predict(X_val_img)
acc_score = accuracy_score(y_val, y_pred_val)
print('所用模型在图像矩阵下的准确率：', acc_score)

# 第八步：将模型放在测试集上预测，将结果写回。
print('正在预测测试集标签...')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
with open('out/test.txt', 'w', encoding='utf-8') as file:
    file.write('guid,tag\n')
    for test_file, tag in zip(test_files, y_pred):
        file.write(f'{test_file},{tag}\n')
print('测试集标签预测完成，已存储在 out/test.txt 中。')