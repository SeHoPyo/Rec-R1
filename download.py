from huggingface_hub import hf_hub_download

filepath = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-C4',
    filename='sampled_item_metadata_1M.jsonl',
    repo_type='dataset'
)
print("Downloaded file path:", filepath)

