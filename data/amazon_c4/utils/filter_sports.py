
import json

# 파일 경로 설정
sports_json_path = '/Users/shingeunbang/RLproj/Rec-R1/data/amazon_c4/sports_json/Sports.json'
filtered_review_path = '/Users/shingeunbang/RLproj/Rec-R1/data/amazon_c4/sports_json/filtered_review_Sports_and_Outdoors.jsonl'
output_path = '/Users/shingeunbang/RLproj/Rec-R1/data/amazon_c4/sports_json/Sports_filtered_by_userid.json'

# 1. filtered_review_Sports_and_Outdoors.jsonl에서 user_id 카운트
user_id_count = {}
with open(filtered_review_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            review = json.loads(line)
            user_id = review.get('user_id')
            if user_id:
                user_id_count[user_id] = user_id_count.get(user_id, 0) + 1
        except Exception:
            continue

# 2. 3번 이상 등장하는 user_id만 추출
target_user_ids = {uid for uid, cnt in user_id_count.items() if cnt >= 3}

# 3. Sports.json에서 해당 user_id가 있는 항목만 추출
with open(sports_json_path, 'r', encoding='utf-8') as f:
    sports_data = json.load(f)

filtered_sports = [item for item in sports_data if item.get('user_id') in target_user_ids]

# 4. 결과 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_sports, f, ensure_ascii=False, indent=2)

print(f"총 {len(filtered_sports)}개의 항목이 {output_path}에 저장되었습니다.")
