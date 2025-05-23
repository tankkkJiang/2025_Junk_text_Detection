"""
tank_main.py
快速垃圾文本检测 (character n‑gram + TF‑IDF + LinearSVC)
与原 main.py 的句向量逻辑回归做精度&速度对比
"""

import os, time, re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC          # 速度快、精度高；如需概率换成 LogisticRegression
from sklearn.linear_model import LogisticRegression

# ---------------- 可复用原工具 ----------------
from utils import tokenize_and_remove_stopwords  # 仍然调用原 utils 的清洗+停用符号过滤
DEFAULT_DATA_DIR = 'data'                        # 与 utils/main.py 保持一致
# ------------------------------------------------

def read_data(path):
    with open(path, encoding='utf-8') as f:
        lines = [l.strip().split('\t', 1) for l in f if '\t' in l]
    tag, text = zip(*lines)
    return list(tag), list(text)

# ————————————————————————————————————————————————————————
# 方案 1：TF‑IDF + LinearSVC （推荐）
# ————————————————————————————————————————————————————————
def train_tfidf_svc(x_train, y_train, x_test):
    """返回预测结果、向量化器与模型"""
    # char 1‑3 gram；min_df 去掉极低频，节省稀疏矩阵大小
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), min_df=2)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)

    clf = LinearSVC()              # 训练极快；如需概率可换成 LogisticRegression(max_iter=1000)
    clf.fit(x_train_vec, y_train)
    pred = clf.predict(x_test_vec)
    return pred, vectorizer, clf

# ————————————————————————————————————————————————————————
# 方案 2：沿用原 Word2Vec+句向量+LogReg baseline，便于公平对比
#        （只移植“generate_*”函数，不再做相似矩阵）
# ————————————————————————————————————————————————————————
from gensim.models import Word2Vec
def word2vec_sentence_vec(texts, dim=100):
    model = Word2Vec(sentences=texts, vector_size=dim, window=5, min_count=1, sg=0)
    wv = model.wv
    sent_vecs = []
    for sent in texts:
        emb = np.mean([wv[c] for c in sent if c in wv], axis=0)
        sent_vecs.append(emb if emb.size else np.zeros(dim))
    return np.array(sent_vecs)

def train_w2v_logreg(x_train_tok, y_train, x_test_tok):
    x_train_vec = word2vec_sentence_vec(x_train_tok)
    x_test_vec  = word2vec_sentence_vec(x_test_tok)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(x_train_vec, y_train)
    pred = logreg.predict(x_test_vec)
    return pred

# ————————————————————————————————————————————————————————
def evaluate(name, true_y, pred_y, start_t, end_t):
    print(f'\n==== {name} ====')
    print(f'耗时: {(end_t-start_t):.2f}s')
    print('F1:', f1_score(true_y, pred_y, average="macro"))
    print('混淆矩阵:\n', confusion_matrix(true_y, pred_y))
    print('分类报告:\n', classification_report(true_y, pred_y, digits=3))

if __name__ == "__main__":
    # 1. 读取 & 清洗
    tags, texts = read_data(os.path.join(DEFAULT_DATA_DIR, 'dataset.txt'))
    # 仍用原 utils 的清洗+去停用字符
    texts_clean = tokenize_and_remove_stopwords(texts)

    # 2. 划分训练/测试
    x_train, x_test, y_train, y_test = train_test_split(
        texts_clean, tags, test_size=0.3, random_state=42, stratify=tags)

    # ------- 方案1：TF‑IDF + LinearSVC --------
    t0 = time.time()
    pred_svc, _, _ = train_tfidf_svc(x_train, y_train, x_test)
    evaluate('TF‑IDF char1‑3 + LinearSVC', y_test, pred_svc, t0, time.time())

    # ------- 方案2：Word2Vec + 句均值 + LogReg (baseline) --------
    # 可按需注释掉
    from utils import clean_text  # 方案2 需要纯字符列表
    x_train_tok = x_train  # 已经是 “无停用符号的纯汉字串”
    x_test_tok  = x_test

    t1 = time.time()
    pred_w2v = train_w2v_logreg(x_train_tok, y_train, x_test_tok)
    evaluate('Word2Vec均值 + LogReg (Baseline)', y_test, pred_w2v, t1, time.time())