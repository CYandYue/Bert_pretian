import os
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import pipeline

tokenizer_path = "/home/cy/ruoyu/bert_ws/temp_data/tokenizer/"
model_path = "/home/cy/ruoyu/bert_ws/temp_data/model_18000_round2"

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-18000"))

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

examples = [
    "That 's one of the black marks in our history , in my [MASK].",
    "At the beginning of the First World [MASK], a German force unsuccessfully attacked the Suez Canal.",
    "Cy is a [MASK] who is so handsome",
    "China is developing rapidly [MASK] technology."
    
]

for example in examples:
    for prediction in fill_mask(example):
        # print(f"{prediction['sequence']}, confidence:{prediction['score']}")
        print(f"被遮掩的原句：{example}")
        print(f"完形填空结果：{prediction['sequence']}")
        break
    print("="*50)

