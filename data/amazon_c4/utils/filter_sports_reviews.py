import json
from tqdm import tqdm

def filter_reviews(file_path, valid_parent_asins):
    # 필터링된 리뷰를 저장할 리스트
    filtered_reviews = []
    
    # JSONL 파일 읽기
    count = 0
    unique_keys = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc='필터링 중'):
            try:
                # 각 라인을 JSON 객체로 파싱
                review = json.loads(line.strip())
                count += 1
                
                # 필터링 조건 적용
                # parent_asin이 metadata_sports_filtered.jsonl에 있는 것만 필터링
                # 중복된 리뷰(동일 parent_asin, user_id)가 포함되지 않도록 set을 사용하여 필터링
                unique_key = (review.get('parent_asin'), review.get('text'))

                if (
                    review.get('rating') == 5.0 and
                    review.get('verified_purchase') is True and
                    len(review.get('text', '')) >= 100 and
                    review.get('parent_asin') in valid_parent_asins and
                    unique_key not in unique_keys
                ):
                    filtered_reviews.append(review)
                    unique_keys.add(unique_key)
            except json.JSONDecodeError:
                continue
    
    print(f"총 {count}개 리뷰 중 {len(filtered_reviews)}개가 조건을 만족했습니다.")
    return filtered_reviews

if __name__ == "__main__":
    valid_parent_asins = set()
    metadata_file = 'data/meta_data/metadata_Sports_and_Outdoors_reformatted.jsonl'
    with open(metadata_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            valid_parent_asins.add(item['item_id'])
    
    # 입력 파일 경로
    input_file = 'data/amazon_c4/sports_json/raw_review_Sports_and_Outdoors.jsonl'

    # 출력 파일 경로
    output_file = 'data/amazon_c4/sports_json/filtered_raw_review_Sports_and_Outdoors.jsonl'

    # 리뷰 필터링
    filtered_reviews = filter_reviews(input_file, valid_parent_asins)

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as file:
        for idx, review in enumerate(filtered_reviews, 1):
            file.write(json.dumps(review) + '\n')
            if idx % 10000 == 0 or idx == len(filtered_reviews):
                print(f"{idx}/{len(filtered_reviews)}개 저장 완료")

    print(f'결과가 {output_file}에 저장되었습니다.') 