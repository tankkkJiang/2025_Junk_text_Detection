#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证垃圾文本检测模型
"""

import os
import numpy as np
from collections import Counter
from pathlib import Path
import joblib
from tqdm import tqdm
import csv
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from utils import tokenize_and_remove_stopwords
from gensim.models import Word2Vec  # 导入 Word2Vec

DEFAULT_DATA_DIR = 'data'  # 数据文件夹名称
SAVE_DIR = Path('models')  # 模型保存目录
RESULTS_FILE = 'validation_results2.csv'  # 结果保存文件

# ========= 通用工具 =========
def read_data(path):
    with open(path, encoding='utf-8') as f:
        tag, txt = zip(*[l.strip().split('\t', 1) for l in f if '\t' in l])
    return list(tag), list(txt)

def evaluate(name, y_true, y_pred):
    print(f'\n==== {name} ====')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Macro‑F1:', f1_score(y_true, y_pred, average="macro"))
    print('混淆矩阵:\n', confusion_matrix(y_true, y_pred))
    print('分类报告:\n', classification_report(y_true, y_pred, digits=3))
    return {
        'model_name': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average="macro"),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, digits=3, output_dict=True)
    }

# ========= 加载模型并验证 =========
def validate_model(model_path, test_data_path):
    # 加载模型
    model_data = joblib.load(model_path)
    if isinstance(model_data, tuple):
        embedder, clf = model_data
    else:
        clf = model_data
        embedder = None

    # 读取测试数据
    y_test, X_test_raw = read_data(test_data_path)

    # 预处理测试数据
    X_test = tokenize_and_remove_stopwords(X_test_raw)

    # 根据模型类型转换测试数据
    if isinstance(embedder, Word2Vec):
        X_test_transformed = np.array([word2vec_transform(embedder, text) for text in X_test])
    else:
        X_test_transformed = embedder.transform(X_test)

    # 使用模型进行预测
    y_pred = clf.predict(X_test_transformed)

    # 评估模型性能
    model_name = os.path.basename(model_path).replace('.pkl', '')
    evaluation_results = evaluate(model_name, y_test, y_pred)

    return evaluation_results

def word2vec_transform(w2v_model, text):
    # 将文本转换为 Word2Vec 向量
    words = list(text)
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

def save_results_to_csv(results, file_path):
    # 保存结果到 CSV 文件
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for result in results:
            model_name = result['model_name']
            report = result['classification_report']
            for class_label, metrics in report.items():
                if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                writer.writerow([
                    model_name,
                    class_label,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score'],
                    metrics['support']
                ])

if __name__ == "__main__":
    # 指定测试数据路径
    test_data_path = os.path.join(DEFAULT_DATA_DIR, '带标签短信.txt')  # 替换为你的测试数据集文件名

    # 验证 models 文件夹中的所有模型
    all_results = []
    for model_file in SAVE_DIR.glob('*.pkl'):
        print(f"正在验证模型: {model_file.name}")
        result = validate_model(model_file, test_data_path)
        all_results.append(result)

    # 保存结果到 CSV 文件
    save_results_to_csv(all_results, RESULTS_FILE)
    print(f"所有模型的验证结果已保存到 {RESULTS_FILE}")
