"""
origin_main.py
-------------------------------------------------------
基于“声形相似 + 动态字向量”的垃圾文本检测
新增：
  1) MODE 1/2（单数据集 or 交叉数据集）
  2) 先扫描缺失字符，一次性 batch-update Word2Vec，再生成加权字向量
-------------------------------------------------------
"""

from tqdm import tqdm
import re
from utils import *
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import os
import time

DEFAULT_DATA_DIR = 'data'    # ← 根据需要修改整个数据文件夹名称

# ======== 默认参数（可一键切换）========================
MODE = 1               # 1=单数据集(50/50)；2=交叉数据集
SINGLE_DATA = 'big_dataset.txt'
TRAIN_DATA  = 'big_dataset.txt'
TEST_DATA   = 'dataset.txt'
# ======================================================


def read_data(filename):
    """
    读取原始语料，每行格式 “标签<TAB>文本”
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag  = [data[0] for data in dataset]
    text = [data[1] for data in dataset]
    return tag, text


def divide_dataset(tag, vector):
    """按 50%:50% 划分训练集和测试集"""
    return train_test_split(vector, tag, test_size=0.5, random_state=42)


def clean_text(dataset):
    """文本清洗：移除所有非中英文、数字和空格字符"""
    cleaned_text = []
    for text in tqdm(dataset, desc='Cleaning text'):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        cleaned_text.append(clean.strip())
    return cleaned_text


def tokenize_and_remove_stopwords(dataset):
    """按字符保留中文、去停用符号"""
    stopwords_file = os.path.join(DEFAULT_DATA_DIR, 'hit_stopwords.txt')
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}

    tokenized_text = []
    for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
        cleaned_text = ''.join([c for c in text
                                if c not in stopwords and re.search("[\u4e00-\u9fa5]", c)])
        tokenized_text.append(cleaned_text)
    return tokenized_text


def generate_w2v_vectors(tokenized_text, d=100):
    """训练 Word2Vec 并提取每个字符的初始向量"""
    model = Word2Vec(sentences=tokenized_text, vector_size=d,
                     window=5, min_count=1, sg=0, workers=4)
    wv = model.wv
    w2v_vectors = {}
    for tokens in tqdm(tokenized_text, desc='Generating word vectors'):
        for word in tokens:
            if word not in w2v_vectors:
                w2v_vectors[word] = wv[word]
    return w2v_vectors


def update(w2v_vectors, tokenized_corpus, character, d=100):
    """
    若字符缺失，快速增量训练 Word2Vec 并补齐向量
    —— 每个字符只补一次（调用前请先判断）
    """
    if character in w2v_vectors:
        return
    model = Word2Vec(sentences=tokenized_corpus + [[character]],
                     vector_size=d, window=5, min_count=1, sg=0,
                     workers=4, epochs=3)
    if character in model.wv:
        w2v_vectors[character] = model.wv[character]
    else:
        print(f"[Fallback] Word2Vec 仍未学到字符 `{character}`，随机初始化其向量。")
        w2v_vectors[character] = np.random.uniform(-0.01, 0.01, size=d)


# ------------------- 重点改写部分 ---------------------
def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat,
                          tokenized_corpus, chinese_characters_count,
                          threshold=0.6):
    """
    先批量扫描“缺失字符”→ 一次性 update，再计算加权向量
    """
    total_chars = len(chinese_characters)
    vec_dim = len(next(iter(w2v_vectors.values())))

    # === 1) 预扫描所有需要补齐的字符 ==================================
    t0 = time.time()
    missing = set()
    for i in range(total_chars):
        for j in range(total_chars):
            if sim_mat[i][j] >= threshold:
                c = chinese_characters[j]
                if c not in w2v_vectors:
                    missing.add(c)
    print(f"[Pre-Update] 共发现 {len(missing)} 个字符缺失，需要补齐向量。")

    # === 2) 一次性调用 update() 批量补向量 ============================
    if missing:
        for k, c in enumerate(sorted(missing)):
            if k % 500 == 0:
                print(f"[Pre-Update] {k}/{len(missing)} → 正在补齐字符 `{c}`")
            update(w2v_vectors, tokenized_corpus, c, d=vec_dim)
    print(f"[Pre-Update] 缺失字符向量补齐完毕，用时 {time.time() - t0:.2f}s")

    # === 3) 正式计算每个字符的加权向量 ================================
    char_vectors = {}
    print(f"[Info] 开始生成 {total_chars} 个字符的加权向量...")
    for i in range(total_chars):
        character = chinese_characters[i]
        if i % 500 == 0:
            print(f"[Progress] {i}/{total_chars} - 当前处理字符: `{character}`")

        # 布尔掩码快速获取相似字符索引
        sim_idxs = [j for j, s in enumerate(sim_mat[i]) if s >= threshold]
        if not sim_idxs:          # 无相似字符，直接零向量
            char_vectors[character] = np.zeros(vec_dim)
            continue

        counts = np.array([chinese_characters_count[chinese_characters[j]]
                           for j in sim_idxs])
        vecs   = np.vstack([w2v_vectors[chinese_characters[j]] for j in sim_idxs])

        emb = (counts[:, None] * vecs).sum(axis=0)
        denom = counts.sum() if counts.sum() else 1
        char_vectors[character] = emb / denom

    print(f"[Info] 所有字符加权向量生成完毕，共生成 {len(char_vectors)} 个向量。")
    return char_vectors
# -----------------------------------------------------


def generate_sentence_vectors(texts, char_vectors, d=100):
    """两层动态路由聚合为句向量"""
    sentence_vectors = []
    default_vec = np.zeros(d)

    for text in tqdm(texts, desc='Generating sentence vectors'):
        n = len(text)
        alpha = np.zeros((n, n))
        for i in range(n):
            vi = char_vectors.get(text[i], default_vec)
            for j in range(i, n):
                vj = char_vectors.get(text[j], default_vec)
                s = np.dot(vi, vj) / np.sqrt(d)
                alpha[i, j] = alpha[j, i] = s

        alpha_hat = np.exp(alpha)
        row_sum = alpha_hat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        alpha_hat /= row_sum

        m = np.zeros(d)
        for i in range(n):
            mi = (alpha_hat[i][:, None] * np.vstack(
                  [char_vectors.get(ch, default_vec) for ch in text])).sum(axis=0)
            m += mi
        sentence_vectors.append(m / d)
    return sentence_vectors


# ------------------- 逻辑回归分类 & 评估 ----------------
def spam_classification(train_tags, train_vecs, test_vecs):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.array(train_vecs), np.array(train_tags))
    return clf.predict(test_vecs)


def evaluation(test_tags, predictions):
    cm = confusion_matrix(np.array(test_tags), np.array(predictions))
    print("混淆矩阵:"); print(cm)
    print("分类报告:"); print(classification_report(np.array(test_tags),
                                                np.array(predictions)))


# =========================== 主流程 ===========================
if __name__ == "__main__":
    # ----- 1. 数据读取 / 预处理 --------------------------------
    if MODE == 1:
        tag, text = read_data(os.path.join(DEFAULT_DATA_DIR, SINGLE_DATA))
        tokenized_text = tokenize_and_remove_stopwords(clean_text(text))
        text_for_chars = [''.join(s) for s in tokenized_text]
    else:
        train_tag, train_text = read_data(os.path.join(DEFAULT_DATA_DIR, TRAIN_DATA))
        test_tag,  test_text  = read_data(os.path.join(DEFAULT_DATA_DIR, TEST_DATA))
        train_tokenized = tokenize_and_remove_stopwords(clean_text(train_text))
        test_tokenized  = tokenize_and_remove_stopwords(clean_text(test_text))
        text_for_chars  = [''.join(s) for s in train_tokenized]

    # ----- 2. 缓存文件 & 声形相似度 -----------------------------
    os.makedirs('res', exist_ok=True)
    HANZI_CACHE  = 'res/hanzi.txt'
    SIMMAT_CACHE = 'res/similarity_matrix.pkl'

    if os.path.exists(HANZI_CACHE):
        chinese_characters, chinese_characters_count, chinese_characters_code = load_chinese_characters(HANZI_CACHE)
    else:
        chinese_characters, chinese_characters_count, chinese_characters_code = \
            count_chinese_characters(text_for_chars, HANZI_CACHE)
    chinese_characters_count = {k: int(v) for k, v in chinese_characters_count.items()}

    if os.path.exists(SIMMAT_CACHE):
        sim_mat = load_sim_mat(SIMMAT_CACHE)
    else:
        sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)

    # ----- 3. Word2Vec 初始向量 -------------------------------
    if MODE == 1:
        w2v_vectors = generate_w2v_vectors(tokenized_text)
    else:
        w2v_vectors = generate_w2v_vectors(train_tokenized)

    # ----- 4. 加权字向量 & 句向量 ------------------------------
    char_vectors = generate_char_vectors(
        chinese_characters, w2v_vectors, sim_mat,
        tokenized_text if MODE == 1 else train_tokenized,
        chinese_characters_count
    )

    if MODE == 1:
        sentence_vectors = generate_sentence_vectors(tokenized_text, char_vectors)
        tr_vec, te_vec, tr_tag, te_tag = divide_dataset(tag, sentence_vectors)
    else:
        train_vec = generate_sentence_vectors(train_tokenized, char_vectors)
        test_vec  = generate_sentence_vectors(test_tokenized,  char_vectors)
        tr_vec, te_vec, tr_tag, te_tag = train_vec, test_vec, train_tag, test_tag

    # ----- 5. 分类 & 评估 --------------------------------------
    preds = spam_classification(tr_tag, tr_vec, te_vec)
    evaluation(te_tag, preds)