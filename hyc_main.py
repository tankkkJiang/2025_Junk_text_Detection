#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_main.py ── 垃圾文本检测：多模型训练 + 保存 + 多数投票
"""

import os, time, re, sys
import numpy as np
from collections import Counter
from pathlib import Path
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm           import LinearSVC
from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.naive_bayes   import MultinomialNB
from gensim.models         import Word2Vec
from sklearn.model_selection import StratifiedKFold

# ---------- 复用原工具 ----------
from utils import tokenize_and_remove_stopwords
DEFAULT_DATA_DIR = 'data'
# ---------------------------------

SAVE_DIR = Path('models')
SAVE_DIR.mkdir(exist_ok=True)

# ========= 通用工具 =========
def read_data(path):
    with open(path, encoding='utf-8') as f:
        tag, txt = zip(*[l.strip().split('\t', 1) for l in f if '\t' in l])
    return list(tag), list(txt)

def evaluate(name, y_true, y_pred, t0, t1):
    print(f'\n==== {name} ====')
    print(f'耗时   : {(t1-t0):.2f}s')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Macro‑F1:', f1_score(y_true, y_pred, average="macro"))
    print('混淆矩阵:\n', confusion_matrix(y_true, y_pred))
    print('分类报告:\n', classification_report(y_true, y_pred, digits=3))

# ========= 单模型训练函数（可调超参） =========
# ========= 1. TF‑IDF + LinearSVC =========
def tfidf_svc(xtr, ytr, xte, *, ngram=(1,3), min_df=2, C=1.0, seed=42):
    vec = TfidfVectorizer(analyzer='char', ngram_range=ngram, min_df=min_df)
    Xtr, Xte = vec.fit_transform(xtr), vec.transform(xte)

    clf = LinearSVC(C=C, random_state=seed)
    clf.fit(Xtr, ytr)

    y_pred  = clf.predict(Xte)
    y_proba = None                     # SVC 不支持 predict_proba
    return y_pred, y_proba, vec, clf


# ========= 2. TF‑IDF + LogisticRegression =========
def tfidf_lr(xtr, ytr, xte, *,
             ngram=(1,3), min_df=2, C=1.0, seed=42):
    vec = TfidfVectorizer(analyzer='char',
                          ngram_range=ngram,
                          min_df=min_df)
    Xtr = vec.fit_transform(xtr)
    Xte = vec.transform(xte)

    clf = LogisticRegression(max_iter=1000,
                             C=C,
                             random_state=seed)
    clf.fit(Xtr, ytr)

    # 硬标签
    y_pred = clf.predict(Xte)

    # 对齐概率列
    proba_raw     = clf.predict_proba(Xte)                 # shape = (n_samples, n_clf_classes)
    n_samples     = Xte.shape[0]                           # 用 shape[0] 代替 len()
    proba_aligned = np.zeros((n_samples, n_classes))
    for raw_col, lab in enumerate(clf.classes_):
        proba_aligned[:, class_to_idx[lab]] = proba_raw[:, raw_col]
    y_proba = proba_aligned

    return y_pred, y_proba, vec, clf


# ========= 3. HashingVectorizer + SGDClassifier =========
def hashing_sgd(xtr, ytr, xte, *, n_features=2**20,
                alpha=1e-4, seed=42, use_log_loss=False):
    vec = HashingVectorizer(analyzer='char', ngram_range=(1,3),
                            n_features=n_features, alternate_sign=False)
    Xtr, Xte = vec.transform(xtr), vec.transform(xte)

    loss = 'log_loss' if use_log_loss else 'hinge'   # log_loss 才能输出概率
    clf  = SGDClassifier(loss=loss, alpha=alpha, random_state=seed)
    clf.fit(Xtr, ytr)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte) if loss == 'log_loss' else None
    return y_pred, y_proba, vec, clf


# ========= 4. TF‑IDF + MultinomialNB =========
def tfidf_nb(xtr, ytr, xte, *, ngram=(1,5), min_df=1):
    vec = TfidfVectorizer(analyzer='char', ngram_range=ngram, min_df=min_df)
    Xtr, Xte = vec.fit_transform(xtr), vec.transform(xte)

    clf = MultinomialNB()
    clf.fit(Xtr, ytr)

    y_pred  = clf.predict(Xte)
    proba_raw = clf.predict_proba(Xte)
    # proba_aligned = np.zeros((len(Xte), n_classes))
    proba_aligned = np.zeros((Xte.shape[0], n_classes))
    for raw_col, lab in enumerate(clf.classes_):
        proba_aligned[:, class_to_idx[lab]] = proba_raw[:, raw_col]
    y_proba = proba_aligned
    return y_pred, y_proba, vec, clf


# ========= 5. Word2Vec‑avg + LogisticRegression =========
def w2v_avg_lr(xtr_tok, ytr, xte_tok, *, dim=100,
               window=5, sg=1, seed=42):
    sents_tr = [list(s) for s in xtr_tok]
    sents_te = [list(s) for s in xte_tok]

    w2v = Word2Vec(sentences=sents_tr, vector_size=dim, window=window,
                   min_count=1, sg=sg, workers=4, seed=seed)

    def sent_vec(chars):
        vecs = [w2v.wv[c] for c in chars if c in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    Xtr = np.vstack([sent_vec(s) for s in sents_tr])
    Xte = np.vstack([sent_vec(s) for s in sents_te])

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, ytr)

    y_pred  = clf.predict(Xte)
    proba_raw = clf.predict_proba(Xte)
    proba_aligned = np.zeros((Xte.shape[0], n_classes))
    # proba_aligned = np.zeros((len(Xte), n_classes))
    for raw_col, lab in enumerate(clf.classes_):
        proba_aligned[:, class_to_idx[lab]] = proba_raw[:, raw_col]
    y_proba = proba_aligned
    return y_pred, y_proba, w2v, clf

# ========= 主程序 =========
if __name__ == "__main__":
    # 1. 数据读取与预处理
    y_all, X_raw = read_data(os.path.join(DEFAULT_DATA_DIR, 'dataset.txt'))
    class_order = sorted(set(y_all))  # 例如 ['0', '1']
    n_classes = len(class_order)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    X_all = tokenize_and_remove_stopwords(X_raw)

    x_tr, x_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.5, random_state=42, stratify=y_all)

    # 2. 配置要训练的模型列表
    model_grid = [
        #   (训练函数,       参数字典,                          名称)
        (tfidf_svc , {'ngram':(1,3), 'C':1.0, 'seed':42},     'svc_n13_c1_s42'),
        (tfidf_svc , {'ngram':(1,4), 'C':0.5, 'seed':13},     'svc_n14_c05_s13'),
        (tfidf_lr  , {'ngram':(1,3), 'C':2.0, 'seed':7},      'lr_n13_c2_s7'),
        (hashing_sgd, {'n_features':2**19, 'alpha':1e-4, 'seed':99}, 'sgd_f19_s99'),
        (tfidf_nb  , {'ngram':(1,5), 'min_df':1},             'nb_n15_df1'),
        (w2v_avg_lr, {'dim':100, 'seed':1},                   'w2v_d100_s1'),
        (w2v_avg_lr, {'dim':200, 'seed':2},                   'w2v_d200_s2'),
    ]

    all_preds, model_names = [], []

    proba_list, proba_wt    = [], []
    weights = [1.0, 1.0, 0.8, 0.8, 1.2, 1.0, 1.0]  # 自定义/学习
    # 3. 训练 + 评估 + 保存
    for (train_fn, params, name), wt in zip(model_grid, weights):
        print(f'\n>>> 训练模型 {name}')
        t0 = time.time()
        y_pred, y_proba, embedder, clf = train_fn(x_tr, y_tr, x_te, **params)
        joblib.dump((embedder, clf), SAVE_DIR / f'{name}.pkl')
        evaluate(name, y_te, y_pred, t0, time.time())

        all_preds.append(y_pred)
        model_names.append(name)

        # 如果该模型有概率输出就追加进列表
        if y_proba is not None:
            proba_list.append(y_proba)  # shape = (n_samples, n_classes)
            proba_wt.append(wt)

    prob_models = [(fn, params, name)
                   for fn, params, name in model_grid
                   if fn is not tfidf_svc and fn is not hashing_sgd]
    # 例如，tfidf_lr/tfidf_nb/w2v_avg_lr

    n_meta = len(prob_models)
    n_tr = len(x_tr)
    n_te = len(x_te)

    # 构造 OOF 概率矩阵 和 测试集概率矩阵
    meta_oof = np.zeros((n_tr, n_meta, n_classes))
    meta_test = np.zeros((n_te, n_meta, n_classes))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for j, (train_fn, params, name) in enumerate(prob_models):
        # — 5 折 CV 产生 OOF
        for train_idx, val_idx in skf.split(x_tr, y_tr):
            X_tr_f = [x_tr[i] for i in train_idx]
            y_tr_f = [y_tr[i] for i in train_idx]
            X_val_f = [x_tr[i] for i in val_idx]
            # 训练并返回对齐后的概率
            _, y_proba_val, vec, clf = train_fn(X_tr_f, y_tr_f, X_val_f, **params)
            meta_oof[val_idx, j] = y_proba_val  # (len(val_idx), n_classes)

        # — 用全量训练集再预测一次测试集概率
        _, y_proba_te, vec, clf = train_fn(x_tr, y_tr, x_te, **params)
        meta_test[:, j, :] = y_proba_te

    # — 拼特征：每个样本有 n_meta * n_classes 个特征，或者只用正类概率 (二分类时)
    # 下面示例取「正类」概率作为特征，若多分类可直接用 one-hot flatten
    meta_oof_feat = meta_oof[:, :, 1]  # shape = (n_tr, n_meta)
    meta_test_feat = meta_test[:, :, 1]  # shape = (n_te, n_meta)

    # — 训练二级模型（Meta-classifier）
    meta_clf = LogisticRegression(max_iter=1000, random_state=0)
    meta_clf.fit(meta_oof_feat, y_tr)

    # — 预测
    y_stack = meta_clf.predict(meta_test_feat)

    # — 评估
    evaluate(f'Stacking (LogReg meta)', y_te, y_stack, 0, 0)


    # ---------- 多数投票 ----------
    all_preds = np.array(all_preds)  # (n_models, n_samples)
    y_vote = np.apply_along_axis(
        lambda col: Counter(col).most_common(1)[0][0], 0, all_preds)
    evaluate(f'Majority Voting ({len(model_names)})', y_te, y_vote, 0, 0)

    # ---------- 软投票（加权平均） ----------
    if proba_list:  # 至少一个模型支持概率
        prob_mat = np.stack(proba_list, axis=0)  # (n_prob_models, n_samples, n_classes)
        wt_arr = np.array(proba_wt).reshape(-1, 1, 1)
        proba_avg = (prob_mat * wt_arr).sum(axis=0) / wt_arr.sum()
        y_soft_idx = proba_avg.argmax(axis=1)  # 先得到整型索引
        y_soft = [idx_to_class[i] for i in y_soft_idx]  # ← 转成字符串标签
        evaluate('Soft Voting (weighted)', y_te, y_soft, 0, 0)
