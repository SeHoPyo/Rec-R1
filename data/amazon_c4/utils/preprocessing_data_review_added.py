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
REVIEWS_JSONL = "data/amazon_c4/sports_json/filtered_raw_review_Sports_and_Outdoors.jsonl"
OUTPUT_DIR = "data/amazon_c4/sports_parquet_review_added"
META_DIR = "data/meta_data/raw_metadata_Sports_and_Outdoors_reformatted.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = """You are an expert in query rewriting for dense retrieval systems. Rewrite the following product search query as if you are a real customer writing a natural, authentic review after using the product. Maintain the meaning and details of the original query, but shift the tone to be more casual, emotional, and based on personal experience. Include specific comments about product performance that match the query's intent.

# Below is the product search query:
# ```{user_query}```"""

PROMPT_WITH_HISTORY = """You are an expert in rewriting product search queries into customer-like product reviews optimized for dense retrieval systems.

# Follow these steps:
# 1. Analyze the user's purchase history, including previous reviews, to identify the features they value, their tone, vocabulary, and writing style.
# 2. Rewrite the provided product search query into an authentic review, as if you have personally used the product.
# 3. Ensure that your review captures all the details from the original query and fully reflects the user's intent.

# Below are the user's purchase history:
# ```{previous_reviews}```

# Below is the product search query:
# ```{user_query}```

# Remember: your main goal is to write a review that fully reflects the product search query, making it sound as if it was personally written by the user.
"""


def load_meta_data():
    """Load meta data from jsonl file"""
    meta_data = {}
    with open(META_DIR, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading meta data"):
            line = json.loads(line.strip())
            meta_data[line['item_id']] = line['metadata']
    return meta_data

def load_user_reviews(meta_data_dict=None):
    """
    Load all reviews and organize them by user_id.
    Optionally, include metadata for each reviewed item if meta_data_dict is provided.
    """
    user_reviews = {}
    review_count = 0
    with open(REVIEWS_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line.strip())
                user_id = review.get('user_id')
                item_id = review.get('parent_asin')
                if user_id:
                    if user_id not in user_reviews:
                        user_reviews[user_id] = []
                    # Prepare review info
                    review_info = {
                        'item_id': item_id,
                        'text': review.get('text', ''),
                    }
                    # Add metadata if available and requested
                    if meta_data_dict is not None and item_id in meta_data_dict:
                        review_info['metadata'] = meta_data_dict[item_id]
                    user_reviews[user_id].append(review_info)
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

def make_prefix(dp, user_reviews_dict, threshold=1024):
    user_id = dp.get('user_id')
    item_id = dp.get('item_id')
    # Start with base prompt
    if user_id:
        # Get previous reviews incrementally, checking threshold each time
        reviews_by_user = user_reviews_dict.get(user_id, [])
        other_reviews = [r for r in reviews_by_user if r['item_id'] != item_id]
        
        # Add reviews one by one until threshold would be exceeded
        formatted_reviews = []
        review_length = 0
        for i, review in enumerate(other_reviews):
            truncated_review = truncate_text(review.get('text', ''))
            truncated_meta_data = truncate_text(review.get('metadata', ''))
            review_item_id = review.get('item_id')
            formatted_review = f"Purchase history {i+1}. \nItem ID: {review_item_id}, Metadata: {truncated_meta_data} \n Previous review: {truncated_review}"
            # Check if adding this review would exceed threshold
            review_length += len(formatted_review.split())
            if review_length < threshold - 200:  # Leave 200 words buffer for the rest of prompt
                formatted_reviews.append(formatted_review)
            else:
                break
        
        # Use the appropriate prompt template
        if formatted_reviews:
            previous_reviews = "\n".join(formatted_reviews)
            input_str = PROMPT_WITH_HISTORY.format(previous_reviews=previous_reviews, user_query=dp['query'])
        else:
            input_str = PROMPT.format(user_query=dp['query'])
    else:
        input_str = PROMPT.format(user_query=dp['query'])
    
    input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
    input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<answer>
{
    "review": xxx
}
</answer>.<|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    
    return input_str

if __name__ == '__main__':
    # Load meta data
    print("Loading meta data from JSONL file...")
    meta_data_dict = load_meta_data()
    print(f"Loaded {len(meta_data_dict)} meta data")

    # Load user reviews first
    print("Loading user reviews from JSONL file...")
    user_reviews_dict = load_user_reviews(meta_data_dict)
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
    threshold = 1024
    
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
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, desc="Processing train split")
    print("Applying processing to validation dataset...")
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True, desc="Processing val split")
    print("Applying processing to test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, desc="Processing test split")
    
    # Calculate statistics about review history
    def count_history(dataset):
        with_history = sum(1 for item in dataset if item['extra_info']['has_review_history'])
        total = len(dataset)
        return with_history, total - with_history
    
    train_with_history, train_without_history = count_history(train_dataset)
    val_with_history, val_without_history = count_history(val_dataset)
    test_with_history, test_without_history = count_history(test_dataset)
    try:
        print("User review history statistics:")
        print(f"Train: {train_with_history} with history ({train_with_history/len(train_dataset)*100:.2f}%), " 
            f"{train_without_history} without ({train_without_history/len(train_dataset)*100:.2f}%)")
        print(f"Val: {val_with_history} with history ({val_with_history/len(val_dataset)*100:.2f}%), "
            f"{val_without_history} without ({val_without_history/len(val_dataset)*100:.2f}%)")
        print(f"Test: {test_with_history} with history ({test_with_history/len(test_dataset)*100:.2f}%), "
            f"{test_without_history} without ({test_without_history/len(test_dataset)*100:.2f}%)")
    except:
        print("Error in counting history")
                
    # Save as parquet
    train_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(OUTPUT_DIR, 'test.parquet'))

    print(f"Saved splits to {OUTPUT_DIR}")
