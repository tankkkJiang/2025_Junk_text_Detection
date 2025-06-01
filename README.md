# 垃圾文本检测

## 1. 系统概述

本系统专注于中文文本处理与分类，借助字音和字形的相似性，构建了一套全面的中文文本分析框架。系统包含汉字编码、相似性计算、文本预处理、特征向量生成以及分类评估等多个模块。

## 2. 代码说明

### 2.1 代码架构
```bash
2025_Junk_text_Detection/
├─ data/                            # 存放所有外部数据文件
│  ├─ chinese_unicode_table.txt     # 汉字 Unicode 编码与笔画等信息对照表
│  ├─ dataset.txt                   # 带标签的垃圾／正常短信语料集（标签\t文本）
│  ├─ hanzi.txt                     # 语料中出现的汉字及其拼音／笔画编码与出现次数
│  ├─ hanzijiegou_2w.txt            # 汉字结构（部件）对照表，用于生成字形编码
│  └─ hit_stopwords.txt             # 停用符号／标点列表，用于分词过滤
├─ four_corner_method/              # 「四角号码」编码方法的实现，供 utils.py 调用
├─ main.py                          # 主脚本：数据读取→清洗→分词→字嵌入→句向量→训练分类器
├─ ssc_similarity.py                # 声形码相似度计算：字音/字形编码相似性函数
├─ utils.py                         # 工具集：汉字编码、统计、相似矩阵构建与加载等
└─ README.md                        # 项目说明与使用指南
```

### 2.2 核心功能文件

`main.py`把所有训练样本的「句向量」当作特征（train_word_vectors），对应的标签当作目标，喂给 LogisticRegression 做训练； 测试时也是把测试样本的句向量（test_word_vectors）输入到同一个训练好的模型里，输出预测标签。
预测标签就是分类器对每条短信判断后的类别标识，0：表示“正常”短信；1：表示“垃圾”短信。

流程：字级别 Word2Vec → 声形相似度加权 → 动态路由句向量 → LogReg

- `ssc_similarity.py`：负责计算汉字之间的字音和字形相似性。它会综合字音编码和字形编码的相似度，得出最终的相似性得分。
- `utils.py`：包含一系列实用工具函数，像汉字编码生成、统计汉字出现次数、构建并加载相似性矩阵等操作都由其完成。
- `four_corner_method/__init__.py`：这是一个独立的模块，用于查询汉字的四角编码，其内部预先加载了四角编码数据。
- `main.py`：作为项目的主程序，它整合了数据读取、文本预处理、特征向量生成、模型训练以及分类评估等全流程操作


### 2.3 数据文件夹

- `data/`：存放训练和测试所需的数据集。

## 3. 安装与使用指南

### 3.1 环境依赖

- Python 3.7 及以上版本
- 所需 Python 库
  ```plaintext
  numpy
  pandas
  scikit-learn
  gensim
  pypinyin
  tqdm
  ```

### 3.2 安装步骤

1. 克隆本项目仓库
2. 安装必要的依赖库

```bash
pip install -r requirements.txt
```

## 4. 改进

### 4.1 缓存声形相似度矩阵

在运行`main.py`过程中可以发现运行下述两步时：
1. Counting characters：遍历所有文本，为每一行里的每个汉字累加出现次数。
2. Computing Character Code：对上一步统计出来的每个不同汉字，逐个调用 ChineseCharacterCoder().generate_character_code（拼音＋四角码＋笔画码），并把结果写入 hanzi.txt。

因为汉字种类只有几千个，这一步在 CPU 上要跑 O(#chars × 编码开销) 大约十几分钟。

我们可以尝试把中间产物（hanzi.txt 和 similarity_matrix.pkl）都存到 res/ 目录下，后续每次运行时：
1. 如果 res/hanzi.txt 已存在，就直接调用 load_chinese_characters 读入，跳过“Counting”+“Computing Character Code”；
2. res/similarity_matrix.pkl 已存在，就直接 load_sim_mat 读入，跳过“compute_sim_mat”；

只在这些缓存文件不存在时，才调用原来的耗时函数并写入缓存。

### 4.2 提出新模型

## 5. 过程记录
```bash
# MODE = 1 表示“单数据集模式”：对 SINGLE_DATA 做 50/50 划分训练/测试
# MODE = 2 表示“交叉数据集模式”：用 TRAIN_DATA 训练，用 TEST_DATA 测试
MODE = 1

# 如果 MODE == 1，脚本会使用 SINGLE_DATA 做划分
SINGLE_DATA = 'big_dataset.txt'

# 如果 MODE == 2，脚本会用 TRAIN_DATA 做训练，用 TEST_DATA 做测试
TRAIN_DATA = 'big_dataset.txt'
TEST_DATA = 'dataset.txt'
```
### 5.1 源代码：原始数据集 `dataset.txt` mode1: 55划分
```bash
[10:22:58] [INFO] 读取数据: data/dataset.txt
[10:22:58] [INFO] ✔︎ 共读取 16007 行样本
[10:22:58] [INFO] 训练样本=8003，测试样本=8004

混淆矩阵：
[[2014  486]
 [ 211 5293]]
[10:28:08] [INFO] 
分类报告：
              precision    recall  f1-score   support

           0      0.905     0.806     0.852      2500
           1      0.916     0.962     0.938      5504

    accuracy                          0.913      8004
   macro avg      0.911     0.884     0.895      8004
weighted avg      0.913     0.913     0.911      8004

[10:28:08] [INFO] 🎉  任务完成！
[10:28:08] [INFO] 🔔 脚本总耗时：0小时05分09秒
```

![](media/2025-06-01-10-25-16.png)
![](media/2025-06-01-10-29-31.png)

### 5.2 源代码：原始数据集 `big_dataset.txt` mode1: 55划分
```bash
[10:30:51] [INFO] 读取数据: data/big_dataset.txt
[10:30:53] [INFO] ✔︎ 共读取 799998 行样本
[10:30:53] [INFO] 训练样本=399999，测试样本=399999

[10:31:30] [INFO] ✔︎ 加权字向量生成完成，大小=4700
Generating sentence vectors: 100%|█████| 399999/399999 [11:09<00:00, 597.66it/s]
Generating sentence vectors: 100%|█████| 399999/399999 [11:07<00:00, 599.45it/s]
[10:53:50] [INFO] 
混淆矩阵：
[[356075   3924]
 [ 10607  29393]]
[10:53:51] [INFO] 
分类报告：
              precision    recall  f1-score   support

           0      0.971     0.989     0.980    359999
           1      0.882     0.735     0.802     40000

    accuracy                          0.964    399999
   macro avg      0.927     0.862     0.891    399999
weighted avg      0.962     0.964     0.962    399999

[10:53:51] [INFO] 🎉  任务完成！
[10:53:51] [INFO] 🔔 脚本总耗时：0小时23分00秒
```

![](media/2025-06-01-10-35-12.png)
![](media/2025-06-01-10-54-07.png)

### 5.3 源代码：mode2: `big_dataset.txt`训练，`dataset.txt`测试
```bash
```

### 5.4 新模型：原始数据集 `dataset.txt` mode1: 55划分
```bash
```

### 5.5 新模型：原始数据集 `big_dataset.txt` mode1: 55划分
```bash
```


### 5.6 新模型：mode2: `big_dataset.txt`训练，`dataset.txt`测试
```bash
```