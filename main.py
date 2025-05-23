"""
main.py
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
    train_vector, test_vector, train_tag, test_tag = train_test_split(vector, tag, test_size=0.5, random_state=42)

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
        cleaned_text = ''.join([char for char in text if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)])
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

def update(w2v_vectors, text, character, d=100):
    """
    动态更新：若新字符不在原模型中，则重新训练 Word2Vec 获取其向量
    仅在 generate_char_vectors 内部调用
    """
    model = Word2Vec(sentences=text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv
    w2v_vectors[character] = word_vectors[character]

    return w2v_vectors

def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count, threshold=0.6):
    """
    基于声形相似度矩阵为每个字符生成加权向量
    chinese_characters       : 字符列表
    w2v_vectors             : 字->初始向量 dict
    sim_mat                 : 声形相似度矩阵
    text                    : 原始文本列表，用于 update
    chinese_characters_count: 频次 dict
    threshold               : 相似度阈值（>= 才加入聚合）
    返回                   : dict，字符->加权向量
    """
    char_vectors = {}
    for i in tqdm(range(len(chinese_characters)), desc='Generating char vectors'):
        character = chinese_characters[i]
        similar_group = []
        for j in range(len(sim_mat[i])):
            if sim_mat[i][j] >= threshold:
                similar_group.append(chinese_characters[j])
        sum_count = 0
        emb = np.zeros_like(w2v_vectors[list(w2v_vectors.keys())[0]])  # 初始化一个全零向量
        for c in similar_group:
            if c not in w2v_vectors.keys():
                update(w2v_vectors, text, c)
            emb += chinese_characters_count[c] * w2v_vectors[c]
            sum_count += chinese_characters_count[c]
        emb /= sum_count if sum_count else 1  # 避免除以0
        char_vectors[character] = emb

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
    for text in tqdm(texts, desc='Generating sentence vectors'):
        alpha = np.zeros((len(text), len(text)))
        for i in range(len(text)):
            for j in range(len(text)):
                alpha[i][j] = alpha[j][i] = np.dot(char_vectors[text[i]], char_vectors[text[j]]) / np.sqrt(d)
        
        alpha_hat = np.zeros_like(alpha)
        for i in range(len(text)):
            for j in range(len(text)):
                alpha_hat[i][j] = alpha_hat[j][i] = np.exp(alpha[i][j]) / np.sum(alpha[i])
        
        m = np.zeros((d,))  # 初始化一个全零向量
        for i in range(len(text)):
            mi = np.zeros((d,))
            for j in range(len(text)):
                mi += alpha_hat[i][j] * char_vectors[text[j]]
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
    tag, text = read_data(os.path.join(DEFAULT_DATA_DIR, 'dataset.txt'))

    # 逐字统计并编码，输出到 hanzi.txt；基于声形码计算相似度矩阵
    chinese_characters, chinese_characters_count, chinese_characters_code = \
        count_chinese_characters(text, os.path.join(DEFAULT_DATA_DIR, 'hanzi.txt'))
    sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)
    
    #text_train, text_test, tag_train, tag_test = divide_dataset(tag, text)

    cleaned_text = clean_text(text)
    tokenized_text = tokenize_and_remove_stopwords(cleaned_text)

    # 训练Word2Vec & 字向量
    w2v_vectors = generate_w2v_vectors(tokenized_text)

    # 基于相似度加权的字向量 & 句向量
    char_vectors = generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count)
    sentence_vectors = generate_sentence_vectors(tokenized_text, char_vectors)
    
    train_vectors, test_vectors, train_tag, test_tag = divide_dataset(tag, sentence_vectors)

    predictions = spam_classification(train_tag, train_vectors, test_vectors)

    evaluation(test_tag, predictions)