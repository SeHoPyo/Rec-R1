import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pdb




# read data/esci/raw/sampled_item_metadata_esci.jsonl
# with open('data/amazon_c4/raw/sampled_item_metadata_1M.jsonl', 'r') as file:
#     sampled_metadata = [json.loads(line) for line in file]


# doc_ids = []

# for item in tqdm(sampled_metadata):
#     doc_ids.append(item['item_id'])

# # save to a numpy to data/esci/raw/esci/doc_ids.npy
# np.save('data/amazon_c4/raw/cache/Amazon-C4', doc_ids)


import pandas as pd
import numpy as np
from tqdm import tqdm

# CSV 파일 읽기
# df = pd.read_csv('/home/s1/sehopyo/Rec-R1/data/data_meta.csv')
# df = pd.read_jsonl('/home/s1/sehopyo/Rec-R1/data/amazon_c4/raw/sport_only.jsonl')

doc_ids = []
with open('/home/s1/sehopyo/Rec-R1/data/amazon_c4/raw/sport_only.jsonl', 'r') as file:
    for line in file:
        item = json.loads(line)
        doc_ids.append(item['item_id'])

# 'item_id' 컬럼 추출
# doc_ids = df['item_id'].tolist()

# numpy로 저장
np.save('data/amazon_c4/raw/cache/Amazon-C4/doc_ids.npy', doc_ids)