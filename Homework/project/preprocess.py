import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
import os
# os.path.abspath(__file__) 作用： 获取当前脚本的完整路径
# os.path.dirname(path) 返回path的父路径
# 可嵌套使用，os.path.dirname(os.path.dirname(path) ) 返回父路径的父路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_path:{}'.format(BASE_DIR))

REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']

def read_stopwords(path):
    """
    返回不重复句子集合
    :param path:
    :return:
    """
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line) # add() 方法用于给集合添加元素，如果添加的元素在集合中已存在，则不执行任何操作。
    return lines


def remove_words(words_list):
    """
    移除指定符合和词语
    :param words_list:
    :return:
    """
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def parse_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)  # 去掉含有缺失值的样本 how=‘any’ :只要有缺失值出现，就删除该行或列
    train_df.fillna('', inplace=True)                             # subset: list在哪些列中查看是否有缺失值 inplace: 是否在原数据上操作
    train_x = train_df.Question.str.cat(train_df.Dialogue)        # str.cat()，将A和B拼接，函数使用的前提是两列的内容都是字符串
    print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word') # 切词，返回list
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))


