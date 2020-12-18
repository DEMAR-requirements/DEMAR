import os
from tqdm import tqdm

path = 'raw_data'

all_samples = dict()
for file in os.listdir('raw_data'):
    project_name = file.split('-')[0]
    project_samples = []
    with open(os.path.join(path, file), 'r', encoding='UTF-8') as f:
        sample = []
        for line in tqdm(f):
            line = line.strip()
            if len(line) > 0:
                sample.append(line)
            else:
                if len(sample) > 0:
                    project_samples.append(sample)
                    sample = []

    all_samples[project_name] = project_samples


document_labelled_samples = dict()
for project_name, project_samples in all_samples.items():
    temp = []
    for sample in project_samples:
        if str(sample[-1]).startswith('class=bug') or str(sample[-1]).startswith('class=0'):
            sample = sample[:-1]
            temp.append((sample, 'non_fr'))
        else:
            if str(sample[-1]).strip().startswith('class'):
                sample = sample[:-1]
            temp.append((sample, 'fr'))

    document_labelled_samples[project_name] = temp

# project_name: [ ([(sentence1, sentence_label), (sentence2, sentence_label)], document_label, project_name), ... ]
sample_list = []
for project_name, document_info in document_labelled_samples.items():
    for (sentences_with_labels, document_label) in document_info:
        inner_document = dict()
        inner_sentences = []
        for sentence_with_label in sentences_with_labels:
            sentence_and_label = dict()
            sentence_label = str(sentence_with_label.split('=')[0]).strip()
            sentence = sentence_with_label[len(sentence_label)+1: ]
            if sentence_label in ['explanation', 'expected', 'current', 'benefit',
                                  'drawback', 'example', 'useless', 'title']:
                sentence_and_label['sentence'] = sentence
                sentence_and_label['sentence_label'] = sentence_label
                inner_sentences.append(sentence_and_label)
        inner_document['sentence_samples'] = inner_sentences
        inner_document['project_name'] = project_name
        inner_document['document_label'] = document_label

        sample_list.append(inner_document)

import json
with open('all.json', 'w') as file_object:
    json.dump(sample_list, file_object)


print('FINISHED')


