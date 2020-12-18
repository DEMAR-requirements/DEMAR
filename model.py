# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import *

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.all_sample_path = '/data/all.json'                                       # 所有样本路径
        self.resplit_dataset = False                                                   # 是否重新划分数据
        self.train_path = 'data/train.json'                                # 训练集
        self.dev_path = 'data/dev.json'                                    # 验证集
        self.test_path = 'data/test.json'                                  # 测试集
        self.document_class_list = self.build_class('data/document_class')        # 文档类别名单
        self.sentence_class_list = self.build_class('data/sentence_class')        # 句子类别名单
        self.save_path = '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_document_classes = len(self.document_class_list)       # 文档类别数
        self.num_sentence_classes = len(self.sentence_class_list)       # 文档类别数
        # self.num_sentence_classes = 100                                 # 句子的最大长度
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.word_pad_size = 32                                         # 每句话处理成的长度(短填长切)
        self.sentence_pad_size = 10                                      # 每个文档处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_hidden_size = 768
        self.dropout = 0.1
        self.rnn_hidden_size = 128
        self.rnn_num_layers = 2

    def build_class(self, path):
        class_dict = dict()
        for x in open(path, 'r').readlines():
            name = x.strip().split('=')[0]
            index = int(x.strip().split('=')[1])
            class_dict[name] = index
        return class_dict


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        """
        bert layer
        """
        # model config (Model,Tokenizer, pretrained weights shortcut
        self.model_config = (BertModel, BertTokenizer, "bert-base-uncased")
        self.bert_tokenizer = self.model_config[1].from_pretrained(self.model_config[2])
        self.bert = self.model_config[0].from_pretrained(self.model_config[2])
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        """
        lstm layer
        """
        self.lstm = nn.LSTM(config.bert_hidden_size, config.rnn_hidden_size, config.rnn_num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        "dropout layer"
        self.dropout = nn.Dropout(config.dropout)
        "document classifier layer"
        self.document_classifier = nn.Linear(config.rnn_hidden_size * 2, 2)
        "sentence classifier layer"
        self.sentence_classifier = nn.Linear(config.rnn_hidden_size * 2, config.sentence_pad_size)

    def forward(self, samples):
        x_list = samples[0]
        mask_list = samples[1]
        bert_hidden_tensor = None
        for index, x in enumerate(x_list):
            x_tensor = torch.tensor(x).to(self.config.device)
            mask_tensor = torch.tensor(mask_list[index]).to(self.config.device)
            # pooled or average???
            poolled_output = self.bert(x_tensor, attention_mask=mask_tensor)[1].unsqueeze(0)  # Batch size 1
            if bert_hidden_tensor is None:
                bert_hidden_tensor = poolled_output
            else:
                bert_hidden_tensor = torch.cat((bert_hidden_tensor, poolled_output), 0)

        # CONTEXT EMBEDDING LAYER
        out, _ = self.lstm(bert_hidden_tensor)
        out = out[:, -1, :]
        out = self.dropout(out)

        # TASK-SPECIFIC LAYER
        document_out = self.document_classifier(out)
        sentence_out = self.sentence_classifier(out)

        return document_out, sentence_out
