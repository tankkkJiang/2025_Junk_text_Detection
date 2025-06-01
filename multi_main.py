"""
multi_main.py
========================================================
垃圾文本检测：多方案快速对比
  - 方案 1  TF-IDF(char1-3)  + LinearSVC      （baseline）
  - 方案 2  TF-IDF(char1-3)  + LogisticReg.   （可输出概率）
  - 方案 3  HashingVectorizer(char1-3) + SGDClassifier
  - 方案 4  TF-IDF(char1-5)  + MultinomialNB  （朴素贝叶斯）
  - 方案 5  Word2Vec(char)   + 平均池化 + LogisticReg.（轻量替代原句向量）
========================================================
用法：
    $ python multi_main.py
========================================================
"""

import os, time, re, sys
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             HashingVectorizer)
from sklearn.svm         import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from gensim.models       import Word2Vec

# ---------- 复用原工具 ----------
from utils import tokenize_and_remove_stopwords
DEFAULT_DATA_DIR = 'data'
# --------------------------------

# MODE = 1 表示“单数据集模式”：对 SINGLE_DATA 做 50/50 划分训练/测试
# MODE = 2 表示“交叉数据集模式”：用 TRAIN_DATA 训练，用 TEST_DATA 测试
MODE = 1

# 如果 MODE == 1，脚本会使用 SINGLE_DATA 做划分
SINGLE_DATA = 'big_dataset.txt'

# 如果 MODE == 2，脚本会用 TRAIN_DATA 做训练，用 TEST_DATA 做测试
TRAIN_DATA = 'big_dataset.txt'
TEST_DATA = 'dataset.txt'
# =====================================


# ========= 通用 I/O =========
# 读取数据集
def read_data(path):
    tag = []
    text = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # 跳过空行
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2 or parts[1].strip() == "":
                # 这行无法拆出“标签”和“正文”，跳过或输出警告
                # 如果想看一下具体是哪些行有问题，可以打印一下：
                print(f"Skipping invalid line: {repr(line)}")
                continue
            t, txt = parts
            tag.append(t)
            text.append(txt)
    return tag, text


def evaluate(name, y_true, y_pred, t0, t1):
    print(f'\n==== {name} ====')
    print(f'耗时: {(t1-t0):.2f}s')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Macro-F1:', f1_score(y_true, y_pred, average="macro"))
    print('混淆矩阵:\n', confusion_matrix(y_true, y_pred))
    print('分类报告:\n', classification_report(y_true, y_pred, digits=3))


# ========= 方案 1: TF-IDF + LinearSVC =========
def tfidf_svc(x_tr, y_tr, x_te, ngram=(1, 3), min_df=2):
    vec = TfidfVectorizer(analyzer='char', ngram_range=ngram, min_df=min_df)
    Xtr = vec.fit_transform(x_tr)
    Xte = vec.transform(x_te)
    clf = LinearSVC()
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


# ========= 方案 2: TF-IDF + LogisticRegression =========
def tfidf_lr(x_tr, y_tr, x_te, ngram=(1, 3), min_df=2):
    vec = TfidfVectorizer(analyzer='char', ngram_range=ngram, min_df=min_df)
    Xtr = vec.fit_transform(x_tr)
    Xte = vec.transform(x_te)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


# ========= 方案 3: HashingVectorizer + SGDClassifier =========
def hashing_sgd(x_tr, y_tr, x_te, n_features=2**20, ngram=(1, 3)):
    vec = HashingVectorizer(analyzer='char', ngram_range=ngram,
                            n_features=n_features, alternate_sign=False)
    Xtr = vec.transform(x_tr)
    Xte = vec.transform(x_te)
    clf = SGDClassifier(loss='hinge')
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


# ========= 方案 4: TF-IDF(char1-5) + MultinomialNB =========
def tfidf_nb(x_tr, y_tr, x_te):
    vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=2)
    Xtr = vec.fit_transform(x_tr)
    Xte = vec.transform(x_te)
    clf = MultinomialNB()
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


# ========= 方案 5: Word2Vec + 平均池化 + LogisticReg. =========
def w2v_avg_lr(x_tr_tokens, y_tr, x_te_tokens,
               dim=100, window=5, min_count=1, sg=1):
    # 将字符串转为字符序列
    sents_tr = [list(s) for s in x_tr_tokens]
    sents_te = [list(s) for s in x_te_tokens]

    w2v = Word2Vec(sentences=sents_tr, vector_size=dim,
                   window=window, min_count=min_count, sg=sg, workers=4)

    def sent_vec(chars):
        vecs = [w2v.wv[c] for c in chars if c in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    Xtr = np.vstack([sent_vec(s) for s in sents_tr])
    Xte = np.vstack([sent_vec(s) for s in sents_te])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_tr)
    return clf.predict(Xte)


# ======================== MAIN =========================
if __name__ == "__main__":
    # 1. 读取 & 预处理
    if MODE == 1:
        # 单数据集模式：对 SINGLE_DATA 做 50/50 划分
        tags, texts = read_data(os.path.join(DEFAULT_DATA_DIR, SINGLE_DATA))
        texts_clean = tokenize_and_remove_stopwords(texts)

        # 2. 训练 / 测试拆分
        x_train, x_test, y_train, y_test = train_test_split(
            texts_clean, tags, test_size=0.5,
            random_state=42, stratify=tags
        )
    else:
        # 交叉数据集模式：TRAIN_DATA 训练，TEST_DATA 测试
        # 先读训练集
        train_tags, train_texts = read_data(os.path.join(DEFAULT_DATA_DIR, TRAIN_DATA))
        train_clean  = tokenize_and_remove_stopwords(train_texts)

        # 再读测试集
        test_tags, test_texts = read_data(os.path.join(DEFAULT_DATA_DIR, TEST_DATA))
        test_clean  = tokenize_and_remove_stopwords(test_texts)

        x_train, y_train = train_clean, train_tags
        x_test,  y_test  = test_clean,  test_tags

    # ---------- 方案 1 ----------
    t0 = time.time()
    pred = tfidf_svc(x_train, y_train, x_test)
    evaluate('TF-IDF char1-3 + LinearSVC', y_test, pred, t0, time.time())

    # ---------- 方案 2 ----------
    t0 = time.time()
    pred = tfidf_lr(x_train, y_train, x_test)
    evaluate('TF-IDF char1-3 + LogisticRegression', y_test, pred, t0, time.time())

    # ---------- 方案 3 ----------
    t0 = time.time()
    pred = hashing_sgd(x_train, y_train, x_test)
    evaluate('HashingVectorizer char1-3 + SGDClassifier', y_test, pred, t0, time.time())

    # ---------- 方案 4 ----------
    t0 = time.time()
    pred = tfidf_nb(x_train, y_train, x_test)
    evaluate('TF-IDF char1-5 + MultinomialNB', y_test, pred, t0, time.time())

    # ---------- 方案 5 ----------
    t0 = time.time()
    pred = w2v_avg_lr(x_train, y_train, x_test)
    evaluate('Word2Vec-avg + LogisticRegression', y_test, pred, t0, time.time())