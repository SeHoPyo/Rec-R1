import json
from tqdm import tqdm

def filter_reviews(file_path):
    # 필터링된 리뷰를 저장할 리스트
    filtered_reviews = []
    
    # JSONL 파일 읽기
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc='필터링 중'):
            try:
                # 각 라인을 JSON 객체로 파싱
                review = json.loads(line.strip())
                count += 1
                
                # 필터링 조건 적용
                if (review.get('rating') == 5.0 and 
                    review.get('verified_purchase') is True and 
                    len(review.get('text', '')) >= 100):
                    filtered_reviews.append(review)
            except json.JSONDecodeError:
                continue
    
    print(f"총 {count}개 리뷰 중 {len(filtered_reviews)}개가 조건을 만족했습니다.")
    return filtered_reviews

# 입력 파일 경로
input_file = 'Rec-R1/data/amazon_c4/sports_json/review_Sports_and_Outdoors.jsonl'

# 출력 파일 경로
output_file = 'Rec-R1/data/amazon_c4/sports_json/filtered_review_Sports_and_Outdoors.jsonl'

# 리뷰 필터링
filtered_reviews = filter_reviews(input_file)

# 결과 저장
with open(output_file, 'w', encoding='utf-8') as file:
    for review in filtered_reviews:
        file.write(json.dumps(review) + '\n')

print(f'결과가 {output_file}에 저장되었습니다.') 