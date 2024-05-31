from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from step02_tokenizer import vocab_size, max_len, truncate_longer_samples

import os
import wandb
os.environ["WANDB_API_KEY"] = "eb393fe9f3e6cddcc91adce1b7f2683e6ca754d6"
os.environ["WANDB_MODE"] = "online"
wandb.init(
    # set the wandb project where this run will be logged
    project="bert_01",

    # track hyperparameters and run metadata
    config={
    "architecture": "bert",
    "dataset": "wikitext2",
    "epochs": 50,
    "batch_size":2,
    }
)


#-----------------function----------------------
# 2 mapping function for corpus pre-process
def encode_with_truncation(exmaple):
    # 如果选择截断，需要将未截断的样本连接起来，成为固定长度为max_len的向量
    return bert_tokenizer(exmaple["sentence"], truncation=True, padding="max_length",
                          max_length=max_len, return_special_tokens_mask=True)

def encode_without_truncation(exmaple):
    return bert_tokenizer(exmaple["sentence"], return_special_tokens_mask=True)

#------------------main--------------------------
dataset_path = "/home/cy/ruoyu/Datasets/bert_data/VALUE_wikitext2_been_done/data"
tokenizer_path = "/home/cy/ruoyu/bert_ws/temp_data/tokenizer/"
old_model_path = "/home/cy/ruoyu/bert_ws/temp_data/model_18000_round1/"
model_path = "/home/cy/ruoyu/bert_ws/temp_data/model"

dataset_train, dataset_test, dataset_val = load_dataset(path=dataset_path, split=["train", "test", "validation"])

# 加载训练好的tokenizer
bert_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

# 对corpus进行tokenize
encode_mapping_func = encode_with_truncation if truncate_longer_samples else encode_without_truncation
tokenized_train  = dataset_train.map(encode_mapping_func, batched=True)
tokenized_test   = dataset_test.map(encode_mapping_func, batched=True)
tokenized_val    = dataset_val.map(encode_mapping_func, batched=True)

if truncate_longer_samples:
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    tokenized_train.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    tokenized_test.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

"""
# 测试例子
x = "my name is cy"
encode_x = bert_tokenizer(x, truncation=True, padding="max_length",
                            max_length=10, return_special_tokens_mask=True)
encode_x_id = encode_x["input_ids"]
decode_x = bert_tokenizer.decode(encode_x["input_ids"])

print(f"原句:{x}")
print(f"词袋编码:{encode_x_id}")
print(f"解码还原:{decode_x}")
"""

# 初始化模型
# 注意这里，max_position_embeddings和tokenizer的max_len有关
original_model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_len)
original_model = BertForMaskedLM(config=original_model_config)

# 初始化BERT的data collator, 随机Mask 20% (default is 15%)的token 
data_collator = DataCollatorForLanguageModeling(tokenizer=bert_tokenizer, mlm=True, mlm_probability=0.2)

# 训练参数
training_args = TrainingArguments(
    output_dir=model_path, 
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=16,
    logging_steps=1000, 
    save_steps=1000,
)
# 加载已经训练的模型
model = BertForMaskedLM.from_pretrained(os.path.join(old_model_path,"checkpoint-18000"))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

