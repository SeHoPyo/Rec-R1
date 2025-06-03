import json
import numpy as np

doc_ids = []
with open('Rec-R1/data/meta_data/metadata_sports_filtered.jsonl', 'r') as file:
    for line in file:
        item = json.loads(line)
        doc_ids.append(item['item_id'])

# numpy로 저장
np.save('Rec-R1/data/meta_data/doc_ids.npy', doc_ids)