"""
file func:
training of tokenizer
"""

import os
import json
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


vocab_size = 30522               # 词表大小
max_len = 512                    # 最大序列长度
truncate_longer_samples = True   # 输入截断处理


special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","<S>","<T>"]
dataset_text_file = "/home/cy/ruoyu/bert_ws/temp_data/text/dataset_train.txt"

tokenizer = BertWordPieceTokenizer()

tokenizer.train(files=dataset_text_file,
                vocab_size=vocab_size,
                special_tokens=special_tokens)

tokenizer.enable_truncation(max_len)

tokenizer_path = "/home/cy/ruoyu/bert_ws/temp_data/tokenizer/"

# 储存词表文件vocab.txt
tokenizer.save_model(tokenizer_path) 

# 储存词表配置文件config.json
with open(os.path.join(tokenizer_path, "config.json"),"w") as f:
    tokenizer_config = {
    "do_lower_case": True,
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "model_max_length": max_len,
    "max_len": max_len,
    }
    json.dump(tokenizer_config,f)
    
    
# 尝试加载训练好的tokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        