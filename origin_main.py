"""
origin_main.py
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

DEFAULT_DATA_DIR = 'data'    # ← 根据需要修改整个数据文件夹名称

# ======== 新增部分：默认参数 =========
# MODE = 1 表示“单数据集模式”：对 SINGLE_DATA 做 50/50 划分、训练与测试
# MODE = 2 表示“交叉数据集模式”：用 TRAIN_DATA 训练，用 TEST_DATA 做测试
MODE = 1

# 如果 MODE == 1，脚本会使用 SINGLE_DATA 做划分
SINGLE_DATA = 'big_dataset.txt'

# 如果 MODE == 2，脚本会用 TRAIN_DATA 做训练，用 TEST_DATA 做测试
TRAIN_DATA = 'big_dataset.txt'
TEST_DATA  = 'dataset.txt'
# =====================================


def read_data(filename):
    """
    读取原始语料，每行格式 “标签<TAB>文本”
    返回：
        tag  ：标签列表，如 ['0','1',...]
        text ：原始文本列表，与标签一一对应
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    return tag, text


def divide_dataset(tag, vector):
    """
    按 50%:50% 划分训练集和测试集
    tag    : 标签列表
    vector : 与标签对应的特征向量列表
    返回：train_vector, test_vector, train_tag, test_tag
    """
    train_vector, test_vector, train_tag, test_tag = train_test_split(
        vector, tag, test_size=0.5, random_state=42
    )

    return train_vector, test_vector, train_tag, test_tag


def clean_text(dataset):
    """
    文本清洗：移除所有非中英文、数字和空格字符
    dataset : 原始文本列表
    返回     : 清洗后的文本列表
    """
    cleaned_text = []
    for text in tqdm(dataset, desc='Cleaning text'):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        cleaned_text.append(clean.strip())
    return cleaned_text


# 停用词处理和文本分割
def tokenize_and_remove_stopwords(dataset):
    """
    分词+去停用符号：逐字符扫描，只保留中文且不在停用词表中的字符
    dataset : 清洗后的文本列表
    返回     : “分词”后（纯汉字串）列表
    """
    stopwords_file = os.path.join(DEFAULT_DATA_DIR, 'hit_stopwords.txt')
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}

    tokenized_text = []
    for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
        cleaned_text = ''.join([
            char for char in text
            if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)
        ])
        tokenized_text.append(cleaned_text)

    return tokenized_text


def generate_w2v_vectors(tokenized_text, d=100):
    """
    训练 Word2Vec 并提取每个字符的初始向量
    tokenized_text : 分词后文本列表（字符序列列表）
    d              : 向量维度
    返回           : dict，字符->向量
    """
    model = Word2Vec(sentences=tokenized_text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv

    w2v_vectors = {}
    for tokens in tqdm(tokenized_text, desc='Generating word vectors'):
        for word in tokens:
            if word not in w2v_vectors.keys():
                w2v_vectors[word] = word_vectors[word]

    return w2v_vectors


def update(w2v_vectors, tokenized_corpus, character, d=100):
    """
    动态更新：若新字符不在原模型中，则重新训练 Word2Vec 获取其向量
    仅在 generate_char_vectors 内部调用
    """
    if character in w2v_vectors:
        return

    # 重新快速训练一次 Word2Vec，确保 character 至少出现一次
    model = Word2Vec(
        sentences=tokenized_corpus + [[character]],  # 保证 character 在语料中
        vector_size=d, window=5, min_count=1, sg=0,
        workers=4, epochs=3
    )

    if character in model.wv:
        w2v_vectors[character] = model.wv[character]
    else:
        # 兜底：随机初始化向量
        print(f"[Fallback] Word2Vec 仍未学到字符 `{character}`，随机初始化其向量。")
        w2v_vectors[character] = np.random.uniform(-0.01, 0.01, size=d)


def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat,
                          tokenized_corpus, chinese_characters_count,
                          threshold=0.6):
    """
    基于声形相似度矩阵为每个字符生成加权向量
    chinese_characters       : 字符列表
    w2v_vectors             : 字->初始向量 dict
    sim_mat                 : 声形相似度矩阵
    tokenized_corpus        : 分词后文本列表（字符序列列表），供 update() 使用
    chinese_characters_count: 频次 dict
    threshold               : 相似度阈值（>= 才加入聚合）
    返回                   : dict，字符->加权向量
    """
    total_chars = len(chinese_characters)
    char_vectors = {}

    print(f"[Info] 开始生成 {total_chars} 个字符的加权向量...")
    for i in range(total_chars):
        character = chinese_characters[i]

        # 每 500 个字符打印一次进度和当前字符
        if i % 500 == 0:
            print(f"[Progress] {i}/{total_chars} - 当前处理字符: `{character}`")

        similar_group = []
        for j in range(total_chars):
            if sim_mat[i][j] >= threshold:
                similar_group.append(chinese_characters[j])

        # 初始化一个全零向量
        emb = np.zeros_like(w2v_vectors[list(w2v_vectors.keys())[0]])
        sum_count = 0

        for c in similar_group:
            if c not in w2v_vectors:
                # 如果某个相似字符不在初始向量中，使用 update() 动态加入并打印信息
                print(f"[Update] 字符 `{c}` 不在初始向量中，调用 update() 重新训练获取向量。")
                update(w2v_vectors, tokenized_corpus, c)

            emb += chinese_characters_count[c] * w2v_vectors[c]
            sum_count += chinese_characters_count[c]

        # 避免除以 0
        if sum_count == 0:
            emb /= 1
        else:
            emb /= sum_count

        char_vectors[character] = emb

    print(f"[Info] 所有字符加权向量生成完毕，共生成 {len(char_vectors)} 个向量。")
    return char_vectors


def generate_sentence_vectors(texts, char_vectors, d=100):
    """
    两层动态路由聚合为句向量（重构论文中的 α̂ 逻辑）
    texts        : 分词后文本列表
    char_vectors : 字->加权向量 dict
    d            : 向量维度
    返回        : 句向量列表
    """
    sentence_vectors = []
    # 默认零向量以防测试集中出现训练集中没有的字符
    default_vec = np.zeros(d)

    for text in tqdm(texts, desc='Generating sentence vectors'):
        alpha = np.zeros((len(text), len(text)))
        for i in range(len(text)):
            for j in range(len(text)):
                if text[i] not in char_vectors:
                    print(f"[Warning] 字符 `{text[i]}` 不在字典中，使用默认向量。")
                if text[j] not in char_vectors:
                    print(f"[Warning] 字符 `{text[j]}` 不在字典中，使用默认向量。")
                vec_i = char_vectors.get(text[i], default_vec)
                vec_j = char_vectors.get(text[j], default_vec)
                alpha[i][j] = alpha[j][i] = np.dot(vec_i, vec_j) / np.sqrt(d)

        alpha_hat = np.zeros_like(alpha)
        for i in range(len(text)):
            denom = np.sum(alpha[i]) if np.sum(alpha[i]) != 0 else 1
            for j in range(len(text)):
                alpha_hat[i][j] = alpha_hat[j][i] = np.exp(alpha[i][j]) / denom

        m = np.zeros((d,))  # 初始化一个全零向量
        for i in range(len(text)):
            mi = np.zeros((d,))
            for j in range(len(text)):
                if text[j] not in char_vectors:
                    # 再次提醒：此字符也不在字典里
                    print(f"[Warning] 字符 `{text[j]}` 不在字典中，使用默认向量。")
                vec_j = char_vectors.get(text[j], default_vec)
                mi += alpha_hat[i][j] * vec_j
            m += mi
        sentence_vectors.append(m / d)

    return sentence_vectors


# 垃圾文本分类
def spam_classification(train_tags, train_word_vectors, test_word_vectors):
    """
    逻辑回归分类
    train_tags          : 训练标签列表
    train_word_vectors  : 训练特征矩阵（列表或 ndarray）
    test_word_vectors   : 测试特征矩阵
    返回                : 预测标签列表
    """
    # 如需平衡类可启用下面两行
    #oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    #train_word_vectors, train_tags = oversampler.fit_resample(train_word_vectors, train_tags)

    logistic_repression = LogisticRegression()
    logistic_repression.fit(np.array(train_word_vectors), np.array(train_tags))

    predictions = logistic_repression.predict(test_word_vectors)
    return predictions


def evaluation(test_tags, predictions):
    # 输出混淆矩阵和分类报告
    cm = confusion_matrix(np.array(test_tags), np.array(predictions))
    print("混淆矩阵:")
    print(cm)

    report = classification_report(np.array(test_tags), np.array(predictions))
    print("分类报告:")
    print(report)


if __name__ == "__main__":
    # 根据 MODE 选择数据加载方式
    if MODE == 1:
        # 单数据集模式：对 SINGLE_DATA 做 50/50 划分
        tag, text = read_data(os.path.join(DEFAULT_DATA_DIR, SINGLE_DATA))
        cleaned_text   = clean_text(text)
        tokenized_text = tokenize_and_remove_stopwords(cleaned_text)
        # 对字符统计和向量生成，使用整个 tokenized_text
        text_for_chars = [''.join(s) for s in tokenized_text]
        tag_for_chars  = tag
    else:
        # 交叉数据集模式：用 TRAIN_DATA 训练，用 TEST_DATA 做测试
        train_tag, train_text = read_data(os.path.join(DEFAULT_DATA_DIR, TRAIN_DATA))
        test_tag,  test_text  = read_data(os.path.join(DEFAULT_DATA_DIR, TEST_DATA))

        # 清洗+分词（训练集 & 测试集）
        train_clean     = clean_text(train_text)
        train_tokenized = tokenize_and_remove_stopwords(train_clean)

        test_clean      = clean_text(test_text)
        test_tokenized  = tokenize_and_remove_stopwords(test_clean)

        # 训练集用于字符统计 & 向量构造
        text_for_chars = [''.join(s) for s in train_tokenized]
        tag_for_chars  = train_tag

    # 创建缓存目录
    os.makedirs('res', exist_ok=True)
    HANZI_CACHE  = os.path.join('res', 'hanzi.txt')
    SIMMAT_CACHE = os.path.join('res', 'similarity_matrix.pkl')

    # ====== 汉字统计 & 编码 ======
    if os.path.exists(HANZI_CACHE):
        chinese_characters, chinese_characters_count, chinese_characters_code = load_chinese_characters(HANZI_CACHE)
    else:
        chinese_characters, chinese_characters_count, chinese_characters_code = count_chinese_characters(
            text_for_chars,  # 原始文本列表（字符串）
            HANZI_CACHE      # 写入 res/hanzi.txt
        )
    # 把频次从字符串转成整数
    chinese_characters_count = {char: int(cnt) for char, cnt in chinese_characters_count.items()}

    # ====== 声形相似度矩阵 ======
    if os.path.exists(SIMMAT_CACHE):
        sim_mat = load_sim_mat(SIMMAT_CACHE)
    else:
        sim_mat = compute_sim_mat(
            chinese_characters,
            chinese_characters_code
            # compute_sim_mat 本身会写入 res/similarity_matrix.pkl
        )

    # ====== Word2Vec & 字向量 ======
    if MODE == 1:
        w2v_vectors = generate_w2v_vectors(tokenized_text)
    else:
        w2v_vectors = generate_w2v_vectors(train_tokenized)

    # ====== 基于相似度加权的字向量 & 句向量 ======
    char_vectors = generate_char_vectors(
        chinese_characters,
        w2v_vectors,
        sim_mat,
        tokenized_text if MODE == 1 else train_tokenized,  # 传入字符序列列表
        chinese_characters_count
    )

    if MODE == 1:
        # 单数据集：直接对 tokenized_text 生成句向量
        sentence_vectors = generate_sentence_vectors(tokenized_text, char_vectors)
        train_vectors, test_vectors, train_tag, test_tag = divide_dataset(tag, sentence_vectors)
    else:
        # 交叉数据集：分别生成训练集 & 测试集的句向量
        train_sentence_vectors = generate_sentence_vectors(train_tokenized, char_vectors)
        test_sentence_vectors  = generate_sentence_vectors(test_tokenized,  char_vectors)
        train_vectors, test_vectors, train_tag, test_tag = (
            train_sentence_vectors,
            test_sentence_vectors,
            train_tag,
            test_tag
        )

    # ====== 分类 & 评估 ======
    predictions = spam_classification(train_tag, train_vectors, test_vectors)
    evaluation(test_tag, predictions)