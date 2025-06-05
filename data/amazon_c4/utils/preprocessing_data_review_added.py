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

# Analyze the user's writing style and priorities from the previous reviews above, then write a new review that embodies those characteristics while addressing the content of the query."""

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
    print(f"Loaded {review_count} total reviews for {len(user_reviews)} users")
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
        return ""
    
    # Filter reviews for different items
    other_reviews = [r for r in user_reviews_dict[user_id] if r['item_id'] != current_item_id]
    
    if not other_reviews:
        return ""
    
    # Sort reviews by length of text (prioritizing longer, more detailed reviews)
    # as they likely contain more style information
    other_reviews.sort(key=lambda x: len(x.get('text', '')), reverse=True)
    
    # Format the reviews with numbers
    formatted_reviews = []
    for i, review in enumerate(other_reviews[:max_reviews]):
        # Truncate very long review texts to avoid token limits
        truncated_text = truncate_text(review.get('text', ''))
        formatted_reviews.append(f"review {i+1}: {truncated_text}")
    
    return "\n\n".join(formatted_reviews)

def make_prefix(dp, user_reviews_dict, threshold=512):
    user_id = dp.get('user_id')
    item_id = dp.get('item_id')
    
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
        formatted_reviews = []
        for i, review in enumerate(other_reviews):
            if i >= 3:  # Still respect max_reviews=3 limit
                break
                
            truncated_text = truncate_text(review.get('text', ''))
            review_length = len(truncated_text.split())
            
            # Check if adding this review would exceed threshold
            if current_length + review_length < threshold - 150:  # Leave 100 words buffer for the rest of prompt
                formatted_reviews.append(f"review {i+1}: {truncated_text}")
                current_length += review_length
                max_reviews_added += 1
            else:
                break
        
        # Use the appropriate prompt template
        if formatted_reviews:
            previous_reviews = "\n\n".join(formatted_reviews)
            input_str = PROMPT_WITH_HISTORY.format(previous_reviews=previous_reviews, user_query=dp['query'])
        else:
            input_str = PROMPT.format(user_query=dp['query'])
    else:
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
    print(f"Found {user_id_counts} items with user_id out of {len(data)} total items")

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
    
    # Define the threshold for prompt length
    threshold = 512
    
    # Create mapping function with review history
    def make_map_fn(split):
        def process_fn(example, idx):
            # Generate the prompt with potential review history
            question = make_prefix(example, user_reviews_dict, threshold)
            
            # Check if this example has user history (for statistics)
            user_id = example.get('user_id')
            has_history = False
            if user_id and user_id in user_reviews_dict:
                item_id = example.get('item_id')
                other_reviews = [r for r in user_reviews_dict[user_id] if r['item_id'] != item_id]
                has_history = len(other_reviews) > 0
            
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
                    'has_review_history': has_history
                }
            }
            return data
        return process_fn
    
    # Apply mapping to all datasets
    print("Applying processing to train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    print("Applying processing to validation dataset...")
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)
    print("Applying processing to test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    # Calculate statistics about review history
    def count_history(dataset):
        with_history = sum(1 for item in dataset if item['extra_info']['has_review_history'])
        total = len(dataset)
        return with_history, total - with_history
    
    train_with_history, train_without_history = count_history(train_dataset)
    val_with_history, val_without_history = count_history(val_dataset)
    test_with_history, test_without_history = count_history(test_dataset)
    
    print("User review history statistics:")
    print(f"Train: {train_with_history} with history ({train_with_history/len(train_dataset)*100:.2f}%), " 
          f"{train_without_history} without ({train_without_history/len(train_dataset)*100:.2f}%)")
    print(f"Val: {val_with_history} with history ({val_with_history/len(val_dataset)*100:.2f}%), "
          f"{val_without_history} without ({val_without_history/len(val_dataset)*100:.2f}%)")
    print(f"Test: {test_with_history} with history ({test_with_history/len(test_dataset)*100:.2f}%), "
          f"{test_without_history} without ({test_without_history/len(test_dataset)*100:.2f}%)")
    
    # Filter by prompt length
    original_train_len = len(train_dataset)
    original_val_len = len(val_dataset)
    original_test_len = len(test_dataset)
    
    # Apply filtering
    train_dataset = train_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    val_dataset = val_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    test_dataset = test_dataset.filter(lambda x: len(x['prompt'][0]['content'].split()) < threshold)
    
    print(f"Final counts after filtering by prompt length:")
    print(f"Train: {len(train_dataset)} (removed {original_train_len - len(train_dataset)})")
    print(f"Val: {len(val_dataset)} (removed {original_val_len - len(val_dataset)})")
    print(f"Test: {len(test_dataset)} (removed {original_test_len - len(test_dataset)})")
    
    # Save as parquet
    train_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'test.parquet'))

    print(f"Saved splits to {OUTPUT_DIR}")
