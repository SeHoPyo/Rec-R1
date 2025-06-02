"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb
import csv
from math import floor
import pandas as pd

PROMPT = """You are an expert in query rewriting for dense retrieval systems. Rewrite the following product search query as if you are a real customer writing a natural, authentic review after using the product. Maintain the meaning and details of the original query, but shift the tone to be more casual, emotional, and based on personal experience. Include specific comments about product performance that match the query's intent.
# Below is the product search query:
# ```{user_query}```"""

# PROMPT = """You are an expert in generating queries for dense retrieval. Given a customer query, your task is to retain the original query while expanding it with additional semantically relevant information, retrieve the most relevant products, ensuring they best meet customer needs. If no useful expansion is needed, return the original query as is.
# Below is the query:
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

# def load_rec_dataset(data_dir, domain_name_list):

#     # train_data_all = []
#     # val_data_all = []

#     with open ('/home/s1/sehopyo/Rec-R1/data/amazon_c4/subset/Sports/filtered_train_Sports.json', 'r') as f:
#         train_data_all = json.load(f)

#     val_data_all = train_data_all[-500:]
#     train_data_all = train_data_all[:-500]

#     test_data_dict = {}
#     for domain_name in domain_name_list:
#         # with open(os.path.join(data_dir, domain_name, f'train.json'), 'r') as f:
#         #     train_data = json.load(f)

#         # train_data_all.extend(train_data[:10000])
#         # split 100 examples from the training data for validation
#         # val_data_all.extend(train_data[10000:10200])

#         with open(os.path.join(data_dir, domain_name, f'test.json'), 'r') as f:
#             test_data_dict[domain_name] = json.load(f)

#     return train_data_all, val_data_all, test_data_dict

def load_rec_dataset(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    total_len = len(data)
    train_end = floor(total_len * 18 / 20)
    val_end = floor(total_len * 19 / 20)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


# 사용 예
data_path = '/home/s1/sehopyo/Rec-R1/data/amazon_c4/subset/Sports/512_sampled_Sports.json'
train_data, val_data, test_data = load_rec_dataset(data_path)

# train_data가 list of dicts 라고 가정
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

# 이제 DataFrame에 대해 to_parquet 사용 가능
train_df.to_parquet('data/amazon_c4/inst/dense/sports/512_train.parquet')
val_df.to_parquet('data/amazon_c4/inst/dense/sports/512_val.parquet')
test_df.to_parquet('data/amazon_c4/inst/dense/sports/512_test.parquet')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--local_dir', default='data/amazon_c4/subset')
#     parser.add_argument('--hdfs_dir', default=None)
#     parser.add_argument('--save_dir', type=str, default='data/amazon_c4/inst/dense/sports')
    
#     args = parser.parse_args()
    
#     domain_name_list = ['Sports']
#     # domain_name_list = ['Video_Games', 'Baby', 'Office', 'Sports']
    
#     data_source = f'amazon_c4' + '_dense'
#     data_source_dict = {ele: f'{data_source}_{ele}' for ele in domain_name_list}

#     file_dir = args.local_dir
#     save_dir = os.path.join(args.save_dir)
#     os.makedirs(save_dir, exist_ok=True)
    
#     train_data, val_data, test_data_dict = load_rec_dataset(file_dir)

#     train_dataset = Dataset.from_list(train_data)
#     val_dataset = Dataset.from_list(val_data)
    
#     test_dataset_dict = {}
#     for domain_name in domain_name_list:
#         test_data = test_data_dict[domain_name]
#         test_dataset_dict[domain_name] = Dataset.from_list(test_data)
    
    
#     def make_map_fn(split, data_source):
#         def process_fn(example, idx):
#             question = make_prefix(example)
#             solution = {
#                 "target": example['item_id'],
#             }
#             data = {
#                 "data_source": data_source + f'_{split}',
#                 "prompt": [{
#                     "role": "user",
#                     "content": question,
#                 }],
#                 "ability": "amazon_review",
#                 "reward_model": {
#                     "style": "rule",
#                     "ground_truth": solution
#                 },
#                 "extra_info": {
#                     'split': split,
#                     'index': idx,
#                 }
#             }
#             return data
#         return process_fn
    
#     train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
#     val_dataset = val_dataset.map(function=make_map_fn('val', data_source), with_indices=True)

#     for domain_name in domain_name_list:
#         test_dataset = test_dataset_dict[domain_name]
#         test_dataset_dict[domain_name] = test_dataset.map(function=make_map_fn('test', data_source_dict[domain_name]), with_indices=True)

#     # concat val_dataset, all test_dataset
#     test_dataset = concatenate_datasets(list(test_dataset_dict.values()))
#     # # shuffle the dataset
#     train_dataset = train_dataset.shuffle(seed=42)
    
#     # concatenate val_1 and val_2
#     val_dataset = concatenate_datasets([val_dataset, test_dataset])
    
#     threshold = 256
    
#     # def truncate(train_dataset, threshold):
#     #     count = 0
#     #     # for those that are exceeding the threshold, we can delete the text between "\nTitle:" to "\nInclusion criteria:"
#     #     for i, d in enumerate(train_dataset):
#     #         if len(d['prompt'][0]['content'].split()) > threshold:
#     #             text = d['prompt'][0]['content']
#     #             count += 1
#     #             words = text.split()
#     #             truncate_length = max(threshold - 200, 0)  # Ensure we don't end up with a negative index
#     #             text = ' '.join(words[-truncate_length:])
#     #             train_dataset[i]['prompt'][0]['content'] = text

#     #     print(f"Truncated {count} examples")
        
#     #     return train_dataset
    
#     # train_dataset = truncate(train_dataset, threshold=threshold)
#     # remove the data that is more than 300 words
#     original_len = len(train_dataset)
#     train_dataset = train_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
#     print(f"Removed {original_len - len(train_dataset)} examples")

#     hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None

#     train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
#     val_dataset.to_parquet(os.path.join(save_dir, 'val.parquet'))
#     test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
    
#     if hdfs_dir is not None:
#         makedirs(hdfs_dir)
#         copy(src=save_dir, dst=hdfs_dir) 
