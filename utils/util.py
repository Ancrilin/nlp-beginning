import jsonlines
from nltk.corpus import stopwords
import time
from datetime import timedelta
from gensim.models import KeyedVectors
import numpy as np
import torch
from tqdm import tqdm


def data_processor(filepath, config, word_to_id=None):
    UNK, PAD = '<UNK>', '<PAD>'
    if word_to_id == None:
        word_to_id = {}
        print('building vocab')
    id_to_word={}
    label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    stop = stopwords.words('english')
    with open(config.stop_path, 'r', encoding='utf-8')as stp:  # 停用词表
        for row in stp:
            stop.append(row.strip())  # 去停用词
    stop_words = set(stop)
    dataset = []
    max_length = float('-inf')
    def get_vocab(t_filepath, t_word_to_id):
        with open(t_filepath, 'r', encoding='utf-8') as fp:
            for i, item in enumerate(jsonlines.Reader(fp)):
                t_sentence1 = item['sentence1'].replace(',', '').replace('.', '').replace('(', '').replace(')',
                                                                                                           '').lower().split(
                    ' ')
                t_sentence2 = item['sentence2'].replace(',', '').replace('.', '').replace('(', '').replace(')',
                                                                                                           '').lower().split(
                    ' ')
                for w in t_sentence1:
                    if w not in stop_words:
                        t_word_to_id[w] = t_word_to_id.get(w, 0) + 1
                for w in t_sentence2:
                    if w not in stop_words:
                        t_word_to_id[w] = t_word_to_id.get(w, 0) + 1
        return word_to_id
    if len(word_to_id.items()) == 0:
        word_to_id = get_vocab(config.train_path, word_to_id)
        word_to_id = get_vocab(config.dev_path, word_to_id)
        word_to_id = get_vocab(config.test_path, word_to_id)
        vocab_list = sorted([_ for _ in word_to_id.items() if _[1] >= config.min_freq], key=lambda x: x[1], reverse=True)
        vocab_dic = {word_count[0]: idx + 2 for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({'<PAD>': 0, '<UNK>': 1})
        word_to_id = vocab_dic
    with open(filepath, 'r', encoding='utf-8') as fp:
        for i, item in enumerate(jsonlines.Reader(fp)):
            label = item['gold_label']
            try:
                label = label_to_id[label]
            except:
                continue
            t_sentence1 = item['sentence1'].replace(',', '').replace('.', '').replace('(', '').replace(')',
                                                                                                       '').lower().split(
                ' ')
            t_sentence2 = item['sentence2'].replace(',', '').replace('.', '').replace('(', '').replace(')',
                                                                                                       '').lower().split(
                ' ')
            sentence1 = []
            sentence2 = []
            for w in t_sentence1:
                if w not in stop_words:
                    sentence1.append(word_to_id[w])
            for w in t_sentence2:
                if w not in stop_words:
                    sentence2.append(word_to_id[w])
            seq_len_sentence1 = len(sentence1)
            seq_len_sentence2 = len(sentence2)
            # 短填长切
            if len(sentence1) < config.pad_size:
                sentence1.extend([word_to_id[PAD]] * (config.pad_size - len(sentence1)))
            else:
                sentence1 = sentence1[:config.pad_size]
                seq_len_sentence1 = config.pad_size
            if len(sentence2) < config.pad_size:
                sentence2.extend([word_to_id[PAD]] * (config.pad_size - len(sentence2)))
            else:
                sentence2 = sentence2[:config.pad_size]
                seq_len_sentence2 = config.pad_size
            dataset.append([label, sentence1, sentence2, seq_len_sentence1, seq_len_sentence2])
    return dataset, word_to_id


def get_pre_embedding(vocab, word2ec_filepath):
    w2v = KeyedVectors.load_word2vec_format(word2ec_filepath, binary=False, encoding='utf-8')
    weight = torch.ones(len(vocab), w2v.vector_size)
    # oov = 0
    for i, word in tqdm(enumerate(vocab.keys())):
        try:
            vector = w2v[word]
        except:
            vector = np.random.rand(w2v.vector_size)
            # oov += 1
        weight[i] = torch.from_numpy(vector)
    # print('oov', oov)
    return weight, w2v.vector_size

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    filepath = '../data/snli_1.0/snli_1.0_dev.jsonl'
    dataset, vocab, label_to_id = data_processor(filepath)
    print(vocab)
    print(label_to_id)
    print(dataset[0])
    id, label, sentence1, sentence2, seq1, seq2 = dataset[0]
    print(sentence1, sentence2)

