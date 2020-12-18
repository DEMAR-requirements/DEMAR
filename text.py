import torch
from transformers import *

list1 = [(1), (2), (3)]
list2 = [(1,2), (1,2), (2,3)]
print(list1+list2)

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          # (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          # (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          # (CTRLModel,       CTRLTokenizer,       'ctrl'),
          # (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          # (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          # (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          # (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          # (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          # (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
         ]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF',
# e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    print(tokenizer)
    model = model_class.from_pretrained(pretrained_weights)
    print(model)

    # Encode text
    string_list = ["Hello World", "I love China", "Hello, my dog is cute"]
    print(tokenizer.tokenize("Hello, my dog is cute"))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Hello, my dog is cute")))
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Hello, my dog is cute")))
    # print(tokenizer.encode("This is the text to encode", add_special_tokens=True))
    print(input_ids)
    print(input_ids.size())
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        print(last_hidden_states)
        print(last_hidden_states.size())

