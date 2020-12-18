# coding: UTF-8
from random import shuffle
import json
import os
import time
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def samples_statistics(project):
    """
    统计基本信息
    :return:
    """
    all_sample = 0
    feature_request = 0
    expected = 0
    current = 0
    benefit = 0
    drawback = 0
    example = 0
    explanation = 0
    useless = 0

    with open('data/all.json', 'r') as f_obj:
        samples = json.load(f_obj)
        for sample in samples:
            sentence_samples = sample['sentence_samples']
            document_label = sample['document_label']
            project_name = sample['project_name']
            if project_name == project:
                all_sample += 1
            else:
                continue
            if document_label == 'fr':
                feature_request += 1
            for index, sentence_info in enumerate(sentence_samples):
                sentence_label = sentence_info['sentence_label']
                if sentence_label == 'expected':
                    expected += 1
                elif sentence_label == 'current':
                    current += 1
                elif sentence_label == 'benefit':
                    benefit += 1
                elif sentence_label == 'drawback':
                    drawback += 1
                elif sentence_label == 'example':
                    example += 1
                elif sentence_label == 'explanation':
                    explanation += 1
                elif sentence_label == 'useless':
                    useless += 1

    print('expected: ', expected)
    print('current: ', current)
    print('benefit: ', benefit)
    print('drawback: ', drawback)
    print('example: ', example)
    print('explanation: ', explanation)
    print('useless: ', useless)
    print('TOTAL', all_sample)
    print(feature_request)

# samples_statistics('hibernate')

def build_dataset(config):
    with open('data/all.json', 'r') as file_obj:
        all_samples = json.load(file_obj)
    if config.resplit_dataset \
            or (not os.path.exists('data/train.json') or not os.path.exists('data/dev.json') or not os.path.exists('data/test.json')):
        split_dataset(all_samples)

    def load_dataset(path, sentence_pad_size, word_pad_size):
        padding_samples = []
        with open(path, 'r') as f_obj:
            samples = json.load(f_obj)
            for sample in samples:
                contents = []
                document_label = sample['document_label']
                sentence_samples = sample['sentence_samples']
                # padding word
                for index, sentence_info in enumerate(sentence_samples):
                    if index == config.sentence_pad_size:
                        break
                    sentence = sentence_info['sentence']
                    sentence_label = sentence_info['sentence_label']
                    sentence_labels_index = config.sentence_class_list[sentence_label]
                    tokens = config.tokenizer.tokenize(sentence)
                    mask = []
                    token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
                    if word_pad_size:
                        if len(token_ids) < word_pad_size:
                            mask = [1] * len(token_ids) + [0] * (word_pad_size - len(token_ids))
                            token_ids += ([0] * (word_pad_size - len(token_ids)))
                        else:
                            mask = [1] * word_pad_size
                            token_ids = token_ids[:word_pad_size]
                    contents.append((token_ids, mask, sentence_labels_index))
                    # padding sentence
                if len(sentence_samples) < sentence_pad_size:
                    for _ in range(0, sentence_pad_size-len(contents)):
                        contents.append(([0] * word_pad_size, [0] * word_pad_size, 0))

                padding_samples.append((contents, config.document_class_list[document_label]))
        return padding_samples
    train = load_dataset(config.train_path, config.sentence_pad_size, config.word_pad_size)
    dev = load_dataset(config.dev_path, config.sentence_pad_size, config.word_pad_size)
    test = load_dataset(config.test_path, config.sentence_pad_size, config.word_pad_size)
    return train, dev, test


def split_dataset(samples):
    """
    split the dataset into train, dev and test (evaluation)
    :return:
    """
    shuffle(samples)
    train = samples[: int(len(samples)*0.8)]
    dev = samples[int(len(samples)*0.8): int(len(samples)*0.9)]
    test = samples[int(len(samples)*0.9):]

    with open('data/train.json', 'w') as file_obj:
        json.dump(train, file_obj)
    with open('data/dev.json', 'w') as file_obj:
        json.dump(dev, file_obj)
    with open('data/test.json', 'w') as file_obj:
        json.dump(test, file_obj)


def build_iterator(dataset, config):
    """
    build batches for training, developing and evaluation
    :param dataset:
    :param config:
    :return:
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        sentences_infos = [_[0] for _ in datas]
        x = []
        mask = []
        sentence_label = []
        for row in sentences_infos:
            x.append([triple[0] for triple in row])
            mask.append([triple[1] for triple in row])
            sentence_label.append([triple[2] for triple in row])

        document_labels = [_[1] for _ in datas]
        y = []
        for index, document_labels in enumerate(document_labels):
            y.append([document_labels] + sentence_label[index])

        return (x, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))