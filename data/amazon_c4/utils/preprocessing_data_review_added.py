"""
Preprocess Sports_filtered3plus.json, split into train/val/test, and save to sports_parquet directory.
Include previous reviews from the same user to provide context.
"""

import os
import json
from datasets import Dataset
from tqdm import tqdm
import random

# Input and output paths
INPUT_JSON = "data/amazon_c4/sports_json/Sports_filtered3plus.json"
REVIEWS_JSONL = "data/amazon_c4/sports_json/filtered_review_Sports_and_Outdoors.jsonl"
OUTPUT_DIR = "data/amazon_c4/sports_parquet_review_added"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = """You are an expert in query rewriting for dense retrieval systems. Rewrite the following product search query as if you are a real customer writing a natural, authentic review after using the product. Maintain the meaning and details of the original query, but shift the tone to be more casual, emotional, and based on personal experience. Include specific comments about product performance that match the query's intent.

# Below is the product search query:
# ```{user_query}```"""

PROMPT_WITH_HISTORY = """You are an expert in query rewriting for dense retrieval systems. Rewrite the following product search query as if you are a real customer writing a natural, authentic review after using the product. Maintain the meaning and details of the original query, but shift the tone to be more casual, emotional, and based on personal experience. Include specific comments about product performance that match the query's intent.

# Below is the product search query:
# ```{user_query}```

# Below are previous reviews written by the same user for other products:
# ```{previous_reviews}```

# Analyze the user's writing style, tone, vocabulary, and review patterns from the previous reviews above.
# Important: Write a new review that MATCHES THE SAME PERSONAL STYLE as the user's previous reviews while addressing the content of the query."""

def load_user_reviews():
    """Load all reviews and organize them by user_id"""
    user_reviews = {}
    review_count = 0
    with open(REVIEWS_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line.strip())
                user_id = review.get('user_id')
                if user_id:
                    if user_id not in user_reviews:
                        user_reviews[user_id] = []
                    # Store review with relevant information
                    user_reviews[user_id].append({
                        'item_id': review.get('asin'),
                        'title': review.get('title', ''),
                        'text': review.get('text', ''),
                        'rating': review.get('rating', 0)
                    })
                    review_count += 1
            except json.JSONDecodeError:
                continue
    print(f"DEBUG: Loaded {review_count} total reviews for {len(user_reviews)} users")
    # Print sample of first 3 users to verify data structure
    sample_users = list(user_reviews.keys())[:3]
    for user in sample_users:
        print(f"DEBUG: Sample user {user} has {len(user_reviews[user])} reviews")
    return user_reviews

def truncate_text(text, max_words=150):
    """Truncate text to a maximum number of words"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + " [...truncated...]"

def get_previous_reviews(user_id, current_item_id, user_reviews_dict, max_reviews=3):
    """Get previous reviews from the same user for different items"""
    if user_id not in user_reviews_dict:
        print(f"DEBUG: User {user_id} not found in reviews dictionary")
        return ""
    
    # Filter reviews for different items
    other_reviews = [r for r in user_reviews_dict[user_id] if r['item_id'] != current_item_id]
    
    if not other_reviews:
        print(f"DEBUG: No other reviews found for user {user_id} aside from current item {current_item_id}")
        return ""
    
    # Sort reviews by length of text (prioritizing longer, more detailed reviews)
    # as they likely contain more style information
    other_reviews.sort(key=lambda x: len(x.get('text', '')), reverse=True)
    
    # Format the reviews - ONLY include text, no title or rating
    formatted_reviews = []
    for review in other_reviews:
        # Truncate very long review texts to avoid token limits
        truncated_text = truncate_text(review.get('text', ''))
        formatted_reviews.append(truncated_text)
    
    print(f"DEBUG: Found {len(formatted_reviews)} previous reviews for user {user_id}")
    return "\n\n".join(formatted_reviews[:max_reviews])

def make_prefix(dp, user_reviews_dict):
    user_id = dp.get('user_id')
    item_id = dp.get('item_id')
    
    print(f"DEBUG: Processing example - user_id: {user_id}, item_id: {item_id}")
    
    # Start with base prompt
    if user_id:
        # Get previous reviews incrementally, checking threshold each time
        all_reviews = []
        reviews_by_user = user_reviews_dict.get(user_id, [])
        other_reviews = [r for r in reviews_by_user if r['item_id'] != item_id]
        
        # Sort reviews by length (longer ones first for more style information)
        other_reviews.sort(key=lambda x: len(x.get('text', '')), reverse=True)
        
        # Calculate base prompt length
        base_prompt = PROMPT.format(user_query=dp['query'])
        base_prompt_length = len(base_prompt.split())
        
        # Test template with one review to estimate overhead
        template_overhead = 0
        if other_reviews:
            test_prompt = PROMPT_WITH_HISTORY.format(
                previous_reviews="Sample review text", 
                user_query=dp['query']
            )
            template_overhead = len(test_prompt.split()) - base_prompt_length - 3  # 3 words for "Sample review text"
        
        current_length = base_prompt_length + template_overhead
        max_reviews_added = 0
        
        # Add reviews one by one until threshold would be exceeded
        for i, review in enumerate(other_reviews):
            if i >= 3:  # Still respect max_reviews=3 limit
                break
                
            truncated_text = truncate_text(review.get('text', ''))
            review_length = len(truncated_text.split())
            
            # Check if adding this review would exceed threshold
            if current_length + review_length < threshold - 100:  # Leave 100 words buffer for the rest of prompt
                all_reviews.append(truncated_text)
                current_length += review_length
                max_reviews_added += 1
                print(f"DEBUG: Added review {i+1} with {review_length} words. Current length: {current_length}")
            else:
                print(f"DEBUG: Skipping review {i+1} as it would exceed threshold. Review length: {review_length}, Current total: {current_length}")
                break
        
        # Use the appropriate prompt template
        if all_reviews:
            previous_reviews = "\n\n".join(all_reviews)
            print(f"DEBUG: Using prompt with {max_reviews_added} reviews for user {user_id}")
            input_str = PROMPT_WITH_HISTORY.format(previous_reviews=previous_reviews, user_query=dp['query'])
        else:
            print(f"DEBUG: Using standard prompt without history for user {user_id}")
            input_str = PROMPT.format(user_query=dp['query'])
    else:
        print(f"DEBUG: No user_id, using standard prompt")
        input_str = PROMPT.format(user_query=dp['query'])
    
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{
    "query": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    
    # Final check of prompt length
    final_length = len(input_str.split())
    print(f"DEBUG: Final prompt length: {final_length} words (threshold: {threshold})")
    
    return input_str

def process_example(example, idx, split, user_reviews_dict):
    question = make_prefix(example, user_reviews_dict)
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
            'user_id': example.get('user_id', None)
        }
    }
    return data

def process_and_filter(split_data, split_name, user_reviews_dict):
    processed = []
    with_history_count = 0
    without_history_count = 0
    
    # Sample a few items to print full examples
    sample_indices = random.sample(range(len(split_data)), min(3, len(split_data)))
    
    for idx, example in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
        item = process_example(example, idx, split_name, user_reviews_dict)
        
        # Check if this example has user history
        user_id = example.get('user_id')
        has_history = False
        if user_id and user_id in user_reviews_dict:
            item_id = example.get('item_id')
            other_reviews = [r for r in user_reviews_dict[user_id] if r['item_id'] != item_id]
            has_history = len(other_reviews) > 0
        
        if has_history:
            with_history_count += 1
        else:
            without_history_count += 1
            
        # Debug print for sample items
        if idx in sample_indices:
            print(f"\nDEBUG: Sample {split_name} item {idx}:")
            print(f"  - User ID: {user_id}")
            print(f"  - Item ID: {example.get('item_id')}")
            print(f"  - Has history: {has_history}")
            print(f"  - Query: {example.get('query')}")
            prompt_length = len(item['prompt'][0]['content'].split())
            print(f"  - Prompt length: {prompt_length} words")
            
        if len(item['prompt'][0]['content'].split()) < threshold:
            processed.append(item)
        else:
            print(f"DEBUG: Skipping item {idx} due to prompt length ({len(item['prompt'][0]['content'].split())} > {threshold})")
    
    print(f"{split_name} statistics:")
    print(f"  - Examples with user history: {with_history_count} ({with_history_count / len(split_data) * 100:.2f}%)")
    print(f"  - Examples without user history: {without_history_count} ({without_history_count / len(split_data) * 100:.2f}%)")
    
    return processed

if __name__ == '__main__':
    # Load user reviews first
    print("Loading user reviews from JSONL file...")
    user_reviews_dict = load_user_reviews()
    print(f"Loaded reviews for {len(user_reviews_dict)} users")
    
    # Check if the file exists
    if not os.path.isfile(REVIEWS_JSONL):
        print(f"ERROR: Reviews file not found at {REVIEWS_JSONL}")
    else:
        print(f"Reviews file exists at {REVIEWS_JSONL}")
    
    # Check if the file exists
    if not os.path.isfile(INPUT_JSON):
        print(f"ERROR: Input data file not found at {INPUT_JSON}")
    else:
        print(f"Input data file exists at {INPUT_JSON}")
        
    # Load input data
    print(f"Loading data from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check for user_id in the data
    user_id_counts = sum(1 for item in data if 'user_id' in item)
    print(f"DEBUG: Found {user_id_counts} items with user_id out of {len(data)} total items")
    
    # Print sample of first 3 items to verify data structure
    for i, item in enumerate(data[:3]):
        print(f"DEBUG: Sample item {i}:")
        print(f"  - user_id: {item.get('user_id', 'MISSING')}")
        print(f"  - item_id: {item.get('item_id', 'MISSING')}")
        print(f"  - query: {item.get('query', 'MISSING')}")

    # Shuffle data for random split
    random.seed(42)
    random.shuffle(data)

    # Split ratios
    n_train = 2048
    n_val = 256
    n_test = 256

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:n_train + n_val + n_test]

    # Process and filter by prompt length
    threshold = 512

    train_processed = process_and_filter(train_data, "train", user_reviews_dict)
    val_processed = process_and_filter(val_data, "val", user_reviews_dict)
    test_processed = process_and_filter(test_data, "test", user_reviews_dict)

    print(f"Final counts after filtering by prompt length:")
    print(f"train: {len(train_processed)}, val: {len(val_processed)}, test: {len(test_processed)}")

    # Save as parquet
    Dataset.from_list(train_processed).to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    Dataset.from_list(val_processed).to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'))
    Dataset.from_list(test_processed).to_parquet(os.path.join(OUTPUT_DIR, 'test.parquet'))

    print(f"Saved splits to {OUTPUT_DIR}")
