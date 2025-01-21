# 多模态情感分析

1. 项目结构
   - `in`
     * `data` 包含所有的文本和图像文件。
     * test_without_label.txt 未预测的测试集文件。
   - `out`
     * test.txt 程序生成的预测标签。
   - `utils`
     * `feature`
       - hybrid.py 用于特征融合的代码。
       - image.py 用于提取图像特征的代码。
       - text.py 用于提取文本特征的代码。
     * classifier.py 分类器模型的代码。
     * data.py 读取文本和图像数据的代码。
   - main.py 程序执行的入口。
   - requirements.txt 项目所需的 Python 包。
   - README.md 本描述文件。

2. 执行方式
   
   **警告：本项目对计算机的内存要求较高，即使使用内存开销最小的方式，若要在完整的数据集上训练和测试，你的计算机至少应该有 8 GB 以上的内存！若要使用实验性功能，对内存的要求会更高！**
   
   (1) 克隆或下载项目文件。
   
   (2) 在终端执行以下命令以安装相应的 Python 包依赖。
   ```bash
   pip install -r requirements.txt
   # pip3 install -r requirements.txt
   ```
   **【注意】** 如果要仅安装 CPU 版的 PyTorch，请先保证电脑可以正常访问 pytorch.org，将 requirements.txt 中的 `torch` 和 `torchvision` 两行去掉，然后在 pytorch.org 按照说明安装对应的包，再执行上述命令安装其他包即可。
   
   (3) 执行程序。
   ```bash
   python main.py
   # python3 main.py
   ```
   按照程序的指示选择对文本和图像提取特征的方式以及对应的分类器（建议使用推荐方式，推荐方式的内存开销少，且用时较快，但是预测结果不一定好）。
   
   **【注意】** 若要使用 BERT 模型，且你的计算机连不上 huggingface 时，请先执行以下命令再运行程序。
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. 自定义数据集
   - 你的数据集应该包含 data 文件夹，train.txt 和 test_without_label.txt，按照本项目中 train.txt 和 test_without_label.txt 的格式填充文件名（guid，不包含扩展名）和标签（在 train.txt 中为 positive、negative 或 netural，在 test_without_label.txt 中为 null）。data 文件夹中文本文件（txt）和图片文件（jpg）的文件名必须与 train.txt 和 test_without_label.txt 一致。
   - 将你的数据集放置在本项目的 in 文件夹中（可以删除原有内容）。
   - 执行 main.py，你的测试集结果会存在 out/test.txt 中。

4. 自定义超参数
   - 你可以自由修改 utils 里的 classifier.py，以及 feature 里的 image.py 和 text.py，调整里面模型的超参数。如果你的计算机内存较小，适当调参可能会让你的计算机勉强带动模型。

5. 参考使用库
   - sklearn：分类、文本特征提取（TF-IDF）
   - skimage：图像特征提取（HOG、HSV）
   - transformers：BERT 提取文本特征
   - torchvision：CNN 提取图像特征
