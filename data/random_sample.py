import json
import random

# 무작위로 선택할 항목 수
n = 4096

# JSON 파일 읽기
with open('/home/s1/sehopyo/Rec-R1/data/amazon_c4/subset/Sports/filtered_train_Sports.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# n개 무작위 선택
sampled_data = random.sample(data, n)

# 리스트를 JSON 파일로 저장 (한 줄씩 쓰는 jsonlines 형식이 아님)
with open(f'/home/s1/sehopyo/Rec-R1/data/amazon_c4/subset/Sports/{n}_sampled_Sports.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)
