"""
file func:
dataset to text.txt for the following training of tokenizer
"""

from datasets import load_dataset

################## functions ###################
def dataset_to_txt(dataset, filename, num = 1000):
    count = 0
    with open(filename, "w") as f:
        for t in dataset["sentence"]:
            print(t, file=f) # 重定向到文件
            count += 1
            if count == num:
                break
        
        print(f"Write {count} sample in {filename}...")
        


dataset_path = "/home/cy/ruoyu/Datasets/bert_data/VALUE_wikitext2_been_done/data"

dataset_train, dataset_test, dataset_val = load_dataset(path=dataset_path, split=["train", "test", "validation"])

# 以txt格式保存，以训练tokenizer
output_file = "/home/cy/ruoyu/bert_ws/temp_data/text/dataset_train.txt"
dataset_to_txt(dataset=dataset_train, filename=output_file, num=len(dataset_train))
