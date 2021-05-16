import argparse
import logging

import chardet
import pypinyin
import os
import random
from datetime import datetime

import numpy as np
import torch
import transformers
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tokenizations.bpe_tokenizer import get_encoder


# 不带声调的(style=pypinyin.NORMAL)
def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


# 带声调的(默认)
def yinjie(word):
    s = ''
    # heteronym=True开启多音字
    for i in pypinyin.pinyin(word, heteronym=True):
        s = s + ''.join(i) + " "
    return s


class logger:
    def __init__(self, path, clevel=logging.DEBUG, Flevel=logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # 设置文件日志
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)


logg = logger('/templat.log', logging.INFO, logging.INFO)


def log(x):
    logg.info(x)


def checkchar(lines):
    # 原文里面的字符串
    chars1 = []
    # small里面的字符串
    chars2 = []
    smalllines = []
    for ch in lines[0]:
        chars1.append(ch)
    chars1 = set(chars1)

    with open('cache/vocab_small.txt', 'r', encoding='utf8') as f:
        small = f.readlines()
        small = [line.replace('\n', ' [SEP] ') for line in small]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        small = ''.join(small)
        smalllines.append(small)

    all_len = len(smalllines)
    for ch in smalllines[0]:
        chars2.append(ch)
    chars2 = set(chars2)

    aa = chars1 - chars2
    bb = chars2 - chars1
    print(len(aa))
    print(len(bb))
    writevocab(chars1)


# 生成词典
def writevocab(chars1):
    with open('cache/vocab_cidian.txt', 'w', encoding='utf8') as f:
        for ch in chars1:
            if len(ch) == 0:
                continue
            if ch.startswith('\\ue'):
                continue
            f.write(ch)
            f.write('\n')
    print('writevocab finish')


def build_files(filename, data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    lines = []
    with open(filename, 'rb') as f:
        text = f.read()
    char = chardet.detect(text)
    print(char['encoding'])
    encoding = char['encoding']
    if 'GB2312' == char['encoding']:
        encoding = 'gb18030'
    try:
        with open(filename, 'r', encoding=encoding) as f:
            nodel = f.readlines()
            nodel = [line.replace('\n', ' [SEP] ') for line in nodel]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
            nodel = ''.join(nodel)
            lines.append(nodel)
            print('reading ok')
    except UnicodeDecodeError as e:
        # os.remove(os.path.join(data_path, filename))
        print('except:', e)
    all_len = len(lines)
    # checkchar(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    for i in tqdm(range(num_pieces)):
        print('开始处理:{}'.format(i))
        parse(i, all_len, lines, tokenized_data_path, num_pieces, full_tokenizer)

    print('finish')


def parse(i, all_len, lines, tokenized_data_path, num_pieces, full_tokenizer):
    sublinesNew = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
    if i == num_pieces - 1:
        sublinesNew.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
    sublines = []
    for line in sublinesNew:
        new = full_tokenizer.tokenize(line)
        new = full_tokenizer.convert_tokens_to_ids(new)
        sublines.append(new)
    full_line = []
    for subline in sublines:
        full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
        full_line.extend(subline)
        full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
    with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w', encoding='utf8') as f:
        for id in full_line:
            f.write(str(id) + ' ')
    print('write-finish：{}'.format(i))


def createfilepath(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)


def createpyfromtemplate(templatename, nodelname):
    newfilename = templatename.replace("template", nodelname)
    alllines = []
    try:
        with open('template/' + templatename, 'r', encoding='utf-8') as f:
            nodel = f.readlines()
            nodel = [line.replace('template', nodelname) for line in nodel]
            nodel = ''.join(nodel)
            alllines.append(nodel)
            print('reading ok')
        with open(nodelname + '/' + newfilename, 'w', encoding='utf8') as f:
            for line in alllines:
                f.write(line)
    except UnicodeDecodeError as e:
        print('except:', e)


def main():
    log('start')
    filename = '全唐诗.txt'
    nodel = filename.split('.')[0]
    nodelname = pinyin(nodel)
    # 新建文件夹
    createfilepath(nodelname)
    createfilepath(nodelname + '/data_' + nodelname)
    createfilepath(nodelname + '/data_' + nodelname+'/tokenized')
    createfilepath(nodelname + '/generated_' + nodelname)
    createfilepath(nodelname + '/model_' + nodelname)
    createfilepath(nodelname + '/model_' + nodelname + '/final_model')
    log('files build')

    # 根据模板创建需要的py文件和bat脚本
    createpyfromtemplate('start-parse-template.bat', nodelname)
    createpyfromtemplate('start-train-template.bat', nodelname)
    createpyfromtemplate('generate_template.py', nodelname)
    createpyfromtemplate('train-template.py', nodelname)

    log('py and bat build')

    # return
    # 根据指定的小说生成训练数据集
    tokenizer_path = 'cache/vocab_small.txt'
    raw_data_path = nodelname + '/data_' + nodelname
    tokenized_data_path = raw_data_path + '/tokenized/'

    from tokenizations import tokenization_bert
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
    full_tokenizer.max_len = 999999

    log('building datas')

    build_files(filename=filename, data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=1,
                full_tokenizer=full_tokenizer, min_length=128)
    log('files built')


if __name__ == '__main__':
    main()
