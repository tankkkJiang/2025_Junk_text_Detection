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

# 读取数据集
def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    return tag, text

# 将数据集划分为训练集和测试集
def divide_dataset(tag, vector):
    train_vector, test_vector, train_tag, test_tag = train_test_split(vector, tag, test_size=0.5, random_state=42)

    return train_vector, test_vector, train_tag, test_tag

# 文本清洗
def clean_text(dataset):
    cleaned_text = []
    for text in tqdm(dataset, desc='Cleaning text'):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        cleaned_text.append(clean.strip())
    return cleaned_text

# 停用词处理和文本分割
def tokenize_and_remove_stopwords(dataset):
    stopwords_file = os.path.join(DEFAULT_DATA_DIR, 'hit_stopwords.txt')
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}

    tokenized_text = []
    for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
        cleaned_text = ''.join([char for char in text if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)])
        tokenized_text.append(cleaned_text)

    return tokenized_text

# 为每个汉字生成初始特征向量
def generate_w2v_vectors(tokenized_text, d=100):
    model = Word2Vec(sentences=tokenized_text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv

    w2v_vectors = {}
    for tokens in tqdm(tokenized_text, desc='Generating word vectors'):
        for word in tokens:
            if word not in w2v_vectors.keys():
                w2v_vectors[word] = word_vectors[word]

    return w2v_vectors

# 为语料库中不曾存在的汉字生成字符向量并动态更新语料库
def update(w2v_vectors, text, character, d=100):
    model = Word2Vec(sentences=text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv
    w2v_vectors[character] = word_vectors[character]

    return w2v_vectors

# 根据字符相似性网络生成最终的字嵌入向量
def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count, threshold=0.6):
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

# 根据字嵌入向量生成句子嵌入向量
def generate_sentence_vectors(texts, char_vectors, d=100):
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

    chinese_characters, chinese_characters_count, chinese_characters_code = \
        count_chinese_characters(text, os.path.join(DEFAULT_DATA_DIR, 'hanzi.txt'))
    sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)
    
    #text_train, text_test, tag_train, tag_test = divide_dataset(tag, text)

    cleaned_text = clean_text(text)
    tokenized_text = tokenize_and_remove_stopwords(cleaned_text)
    w2v_vectors = generate_w2v_vectors(tokenized_text)
    char_vectors = generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count)
    sentence_vectors = generate_sentence_vectors(tokenized_text, char_vectors)
    
    train_vectors, test_vectors, train_tag, test_tag = divide_dataset(tag, sentence_vectors)

    predictions = spam_classification(train_tag, train_vectors, test_vectors)

    evaluation(test_tag, predictions)