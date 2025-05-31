"""
main.py  - 2025-05-31
垃圾文本检测：单/交叉数据集两种模式
========================================================
# MODE = 1  → “单数据集模式”：对 SINGLE_DATA 做 50/50 划分
# MODE = 2  → “交叉数据集模式”：用 TRAIN_DATA 训练，用 TEST_DATA 测试
========================================================
依赖：
  pip install numpy gensim scikit-learn imbalanced-learn tqdm
"""

import os, re, sys, logging
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from gensim.models import Word2Vec

# ---------- 项目自带工具函数 ----------
from utils import (
    count_chinese_characters, load_chinese_characters,
    compute_sim_mat,      load_sim_mat
)

# ---------------- 全局配置 ----------------
DEFAULT_DATA_DIR = 'data'          # 统一的数据根目录
RES_DIR           = 'res'          # 缓存目录
os.makedirs(RES_DIR, exist_ok=True)

# MODE = 1 表示“单数据集模式”：对 SINGLE_DATA 做 50/50 划分训练/测试
# MODE = 2 表示“交叉数据集模式”：用 TRAIN_DATA 训练，用 TEST_DATA 测试
MODE = 1

# 如果 MODE == 1，脚本会使用 SINGLE_DATA 做划分
SINGLE_DATA = 'big_dataset.txt'

# 如果 MODE == 2，脚本会用 TRAIN_DATA 做训练，用 TEST_DATA 做测试
TRAIN_DATA = 'big_dataset.txt'
TEST_DATA = 'dataset.txt'

# ---------------------------------------------------------------

# 日志：INFO 级别即可；可改为 DEBUG 看更细日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ---------------- 基础 I/O ----------------
def read_data(filename: str) -> Tuple[List[str], List[str]]:
    """
    读取原始语料，每行格式： '标签<TAB>文本'
    返回 (tag_list, text_list)
    """
    logging.info('读取数据: %s', filename)
    if not os.path.isfile(filename):
        logging.error('❌ 文件不存在: %s', filename)
        sys.exit(1)

    with open(filename, 'r', encoding='utf-8') as f:
        raw = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in raw]
    dataset = [d for d in dataset if len(d) == 2 and d[1].strip()]
    if not dataset:
        logging.error('❌ 文件内容为空或格式不符: %s', filename)
        sys.exit(1)

    tag, text = zip(*dataset)
    logging.info('✔︎ 共读取 %d 行样本', len(text))
    return list(tag), list(text)


# ---------------- 文本预处理 ----------------
def clean_text(dataset: List[str]) -> List[str]:
    """
    清洗：保留中文 / 英文 / 数字 / 空格
    """
    cleaned = []
    for line in tqdm(dataset, desc='Cleaning text', ncols=80):
        cleaned.append(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', line).strip())
    return cleaned


def tokenize_and_remove_stopwords(dataset: List[str]) -> List[str]:
    """
    逐字符“分词”，仅保留中文，且不在停用词表中的字符
    未找到停用词文件时给出警告并继续
    """
    stop_file = os.path.join(DEFAULT_DATA_DIR, 'hit_stopwords.txt')
    if os.path.isfile(stop_file):
        with open(stop_file, 'r', encoding='utf-8') as f:
            stopwords = {s.strip() for s in f}
        logging.info('✔︎ 停用词加载完成，数量=%d', len(stopwords))
    else:
        stopwords = set()
        logging.warning('⚠️  未找到停用词文件 %s，跳过停用', stop_file)

    tokenized = []
    for line in tqdm(dataset, desc='Tokenizing / filtering', ncols=80):
        tokenized.append(''.join(
            [c for c in line if (c not in stopwords and re.search('[\u4e00-\u9fa5]', c))]
        ))
    return tokenized


# ---------------- Word2Vec ----------------
def train_w2v(sentences: List[str], d: int = 100) -> dict:
    """
    使用 gensim Word2Vec 训练字向量；返回 {字:向量} 字典
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=d,
        window=5,
        min_count=1,
        sg=0,
        workers=os.cpu_count()
    )
    w2v = {}
    for sent in tqdm(sentences, desc='Collecting vectors', ncols=80):
        for ch in sent:
            if ch not in w2v:
                w2v[ch] = model.wv[ch]
    logging.info('✔︎ Word2Vec 训练完成，字表大小=%d', len(w2v))
    return w2v


# ---------------- 加权字向量 ----------------
def generate_char_vectors(
    chinese_chars: List[str],
    w2v: dict,
    sim_mat: np.ndarray,
    char_count: dict,
    threshold: float = 0.6
) -> dict:
    """
    论文里的加权字向量；若某字符在 w2v 中不存在 → 直接跳过
    """
    ref_vec = next(iter(w2v.values()))
    d = ref_vec.shape[0]
    char_vec = {}

    for i in tqdm(range(len(chinese_chars)), desc='Generating char vectors', ncols=80):
        ch = chinese_chars[i]
        if ch not in w2v:
            continue  # OOV 字符直接忽略

        # 找相似字符
        group = [ch2 for j, ch2 in enumerate(chinese_chars) if sim_mat[i][j] >= threshold]
        total = 0
        emb   = np.zeros(d, dtype=np.float32)
        for g in group:
            if g not in w2v:
                continue
            emb   += char_count.get(g, 1) * w2v[g]
            total += char_count.get(g, 1)
        char_vec[ch] = emb / (total if total else 1)

    logging.info('✔︎ 加权字向量生成完成，大小=%d', len(char_vec))
    return char_vec


# ---------------- 句向量 ----------------
def generate_sentence_vectors(texts: List[str], char_vec: dict, d: int = 100) -> List[np.ndarray]:
    """
    动态路由句向量。对 char_vec 中缺失的字符直接跳过
    """
    sent_vecs = []
    for sent in tqdm(texts, desc='Generating sentence vectors', ncols=80):
        chars = [c for c in sent if c in char_vec]
        if not chars:                            # 整句无合法字符
            sent_vecs.append(np.zeros(d, dtype=np.float32))
            continue

        n = len(chars)
        alpha = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                alpha[i, j] = np.dot(char_vec[chars[i]], char_vec[chars[j]]) / np.sqrt(d)

        # softmax
        alpha_hat = np.exp(alpha) / np.sum(np.exp(alpha), axis=1, keepdims=True)

        m = np.zeros(d, dtype=np.float32)
        for i in range(n):
            mi = np.zeros(d, dtype=np.float32)
            for j in range(n):
                mi += alpha_hat[i, j] * char_vec[chars[j]]
            m += mi
        sent_vecs.append(m / d)
    return sent_vecs


# ---------------- 分类 / 评估 ----------------
def spam_classification(train_y, train_X, test_X):
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(np.asarray(train_X), np.asarray(train_y))
    return logreg.predict(test_X)


def evaluate(true_y, pred_y):
    logging.info('\n混淆矩阵：\n%s', confusion_matrix(true_y, pred_y))
    logging.info('\n分类报告：\n%s', classification_report(true_y, pred_y, digits=3))


# ---------------- 主程序 ----------------
def main():
    start_time = time.time()
    # --------------------------------------------------
    # 1) 数据加载
    # --------------------------------------------------
    if MODE == 1:
        # 单数据集 ➜ 50/50
        file_path = os.path.join(DEFAULT_DATA_DIR, SINGLE_DATA)
        tags, texts = read_data(file_path)
        tags_train, tags_test, texts_train, texts_test = train_test_split(
            tags, texts, test_size=0.5, random_state=42, stratify=tags
        )
    else:
        # 交叉数据集
        tr_path = os.path.join(DEFAULT_DATA_DIR, TRAIN_DATA)
        te_path = os.path.join(DEFAULT_DATA_DIR, TEST_DATA)
        tags_train, texts_train = read_data(tr_path)
        tags_test,  texts_test  = read_data(te_path)

    logging.info('训练样本=%d，测试样本=%d', len(texts_train), len(texts_test))

    # --------------------------------------------------
    # 2) 文本清洗 & “分词”
    # --------------------------------------------------
    clean_train = clean_text(texts_train)
    token_train = tokenize_and_remove_stopwords(clean_train)

    clean_test  = clean_text(texts_test)
    token_test  = tokenize_and_remove_stopwords(clean_test)

    # --------------------------------------------------
    # 3) 汉字统计 / 声形相似度
    #    仅基于【训练集】统计，测试集出现的 OOV 会被忽略
    # --------------------------------------------------
    HANZI_CACHE = os.path.join(RES_DIR, 'hanzi.txt')
    SIM_CACHE   = os.path.join(RES_DIR, 'similarity_matrix.pkl')

    if os.path.isfile(HANZI_CACHE):
        chinese_chars, char_count, char_code = load_chinese_characters(HANZI_CACHE)
        logging.info('✔︎ 从缓存加载汉字表，大小=%d', len(chinese_chars))
    else:
        chinese_chars, char_count, char_code = count_chinese_characters(
            texts_train, HANZI_CACHE)
        logging.info('✔︎ 重新统计汉字表并写入缓存')

    # 统一把频次转 int
    char_count = {c: int(n) for c, n in char_count.items()}

    if os.path.isfile(SIM_CACHE):
        sim_mat = load_sim_mat(SIM_CACHE)
        logging.info('✔︎ 从缓存加载声形相似度矩阵')
    else:
        sim_mat = compute_sim_mat(chinese_chars, char_code)
        logging.info('✔︎ 重新计算相似度矩阵并写入缓存')

    # --------------------------------------------------
    # 4) Word2Vec & 向量构建
    # --------------------------------------------------
    w2v = train_w2v(token_train, d=100)
    char_vec = generate_char_vectors(chinese_chars, w2v, sim_mat, char_count, threshold=0.6)

    sent_vec_train = generate_sentence_vectors(token_train, char_vec, d=100)
    sent_vec_test  = generate_sentence_vectors(token_test,  char_vec, d=100)

    # --------------------------------------------------
    # 5) 训练 + 预测 + 评估
    # --------------------------------------------------
    pred = spam_classification(tags_train, sent_vec_train, sent_vec_test)
    evaluate(tags_test, pred)
    logging.info('🎉  任务完成！')

    # 记录脚本结束时间并打印耗时
    end_time = time.time()
    elapsed = end_time - start_time
    # 格式化为 时:分:秒
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    logging.info('🔔 脚本总耗时：%d小时%02d分%02d秒', h, m, s)


if __name__ == '__main__':
    main()