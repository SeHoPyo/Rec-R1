import json

input_file = "/Users/shingeunbang/RLproj/Rec-R1/data/meta_data/sampled_item_metadata_1M.jsonl"
output_file = "/Users/shingeunbang/RLproj/Rec-R1/data/meta_data/metadata_sports_filtered.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        # Check if 'category' exists and contains 'Sports'
        # 'category' can be a string or a list
        category = item.get("category", None)
        if category is None:
            continue
        if isinstance(category, str):
            if "Sports" in category:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif isinstance(category, list):
            if any("Sports" in str(cat) for cat in category):
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
