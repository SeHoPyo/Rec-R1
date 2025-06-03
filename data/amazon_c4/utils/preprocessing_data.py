"""
Preprocess Sports_filtered3plus.json, split into train/val/test, and save to sports_parquet directory.
"""

import os
import json
from datasets import Dataset
from tqdm import tqdm
import random

# Input and output paths
INPUT_JSON = "/Users/shingeunbang/RLproj/Rec-R1/data/amazon_c4/sports_json/Sports_filtered3plus.json"
OUTPUT_DIR = "/Users/shingeunbang/RLproj/Rec-R1/data/amazon_c4/sports_parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = """You are an expert in query rewriting for dense retrieval systems. Rewrite the following product search query as if you are a real customer writing a natural, authentic review after using the product. Maintain the meaning and details of the original query, but shift the tone to be more casual, emotional, and based on personal experience. Include specific comments about product performance that match the query's intent.
# Below is the product search query:
# ```{user_query}```"""

def make_prefix(dp):
    input_str = PROMPT.format(user_query=dp['query'])
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return input_str

def process_example(example, idx, split):
    question = make_prefix(example)
    solution = {
        "target": example.get('item_id', None),
    }
    data = {
        "data_source": "amazon_c4_dense_sports",
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "amazon_review",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
    return data

if __name__ == '__main__':
    # Load input data
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle data for random split
    random.seed(42)
    random.shuffle(data)

    # Split ratios
    n = len(data)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Process and filter by prompt length
    threshold = 256

    def process_and_filter(split_data, split_name):
        processed = []
        for idx, example in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            item = process_example(example, idx, split_name)
            if len(item['prompt'][0]['content'].split()) < threshold:
                processed.append(item)
        return processed

    train_processed = process_and_filter(train_data, "train")
    val_processed = process_and_filter(val_data, "val")
    test_processed = process_and_filter(test_data, "test")

    print(f"train: {len(train_processed)}, val: {len(val_processed)}, test: {len(test_processed)}")

    # Save as parquet
    Dataset.from_list(train_processed).to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    Dataset.from_list(val_processed).to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'))
    Dataset.from_list(test_processed).to_parquet(os.path.join(OUTPUT_DIR, 'test.parquet'))

    print(f"Saved splits to {OUTPUT_DIR}")
