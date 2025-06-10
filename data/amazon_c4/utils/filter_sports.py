import json
from collections import defaultdict

# 파일 경로 설정
sports_json_path = 'data/amazon_c4/sports_json/Sports.json'
filtered_review_path = 'data/amazon_c4/sports_json/filtered_raw_review_Sports_and_Outdoors.jsonl'
output_path = 'data/amazon_c4/sports_json/Sports_filtered3plus.json'

# user_id 등장 횟수 카운트
user_count = defaultdict(int)

# filtered_review 파일에서 user_id 카운트
with open(filtered_review_path, 'r') as f:
    for line in f:
        review = json.loads(line)
        user_id = review.get('user_id')
        if user_id:
            user_count[user_id] += 1

# 3번 이상 등장한 user_id 추출
frequent_users = {user_id for user_id, count in user_count.items() if count >= 3}
print(f"3번 이상 등장한 사용자 수: {len(frequent_users)}")

# Sports.json 파일 필터링
filtered_data = []
with open(sports_json_path, 'r') as f:
    data = json.load(f)
    for item in data:
        if item.get('user_id') in frequent_users:
            filtered_data.append(item)

print(f"원본 데이터 항목 수: {len(data)}")
print(f"필터링된 데이터 항목 수: {len(filtered_data)}")

# 필터링된 데이터 저장
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"필터링된 데이터가 {output_path}에 저장되었습니다.")
