"""
Preprocess Sports_filtered3plus.json, split into train/val/test, and save to sports_parquet directory.
"""

import os
import json
from datasets import Dataset
from tqdm import tqdm
import random

# Input and output paths
INPUT_JSON = "data/amazon_c4/sports_json/Sports_filtered3plus.json"
OUTPUT_DIR = "data/amazon_c4/sports_parquet"
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

if __name__ == '__main__':
    # Load input data
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle data for random split
    random.seed(42)
    random.shuffle(data)

    # Split ratios
    n_train = 1024
    n_val = 128
    n_test = 128

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:n_train + n_val + n_test]

    # Convert to datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Process using map function
    def make_map_fn(split):
        def process_fn(example, idx):
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
        return process_fn
    
    # Apply mapping
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    # Process and filter by prompt length
    threshold = 512
    
    # Filter datasets
    original_train_len = len(train_dataset)
    original_val_len = len(val_dataset)
    original_test_len = len(test_dataset)
    
    train_dataset = train_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    val_dataset = val_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    test_dataset = test_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    
    print(f"train: {len(train_dataset)} (removed {original_train_len - len(train_dataset)})")
    print(f"val: {len(val_dataset)} (removed {original_val_len - len(val_dataset)})")
    print(f"test: {len(test_dataset)} (removed {original_test_len - len(test_dataset)})")

    # Save as parquet
    train_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'test.parquet'))

    print(f"Saved splits to {OUTPUT_DIR}")
