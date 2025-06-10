"""
Script to load the raw_review_Sports_and_Outdoors dataset from the local dataset script
and save the examples to a JSONL file.
"""

import json
from datasets import load_dataset

import json
from datasets import load_dataset
import sys
import multiprocessing as mp

def write_jsonl_chunk(chunk, output_file, start_idx):
    """Write a chunk of data to a JSONL file, appending."""
    with open(output_file, "a", encoding="utf-8") as f:
        for idx, example in enumerate(chunk, start=start_idx):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if idx % 10000 == 0:
                print(f"  ... {idx} records written", flush=True)

def chunked_iterable(iterable, chunk_size):
    """Yield successive chunk_size-sized chunks from iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def parallel_write_jsonl(ds, output_file, num_workers=4, chunk_size=10000):
    # Remove file if exists to avoid appending to old data
    import os
    if os.path.exists(output_file):
        os.remove(output_file)
    total = len(ds)
    print(f"Saving {total} records to {output_file} using {num_workers} workers ...")
    pool = mp.get_context("spawn").Pool(num_workers)
    jobs = []
    start_idx = 1
    for chunk in chunked_iterable(ds, chunk_size):
        jobs.append(pool.apply_async(write_jsonl_chunk, (chunk, output_file, start_idx)))
        start_idx += len(chunk)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()
    print(f"Saved {total} records to {output_file}")

def main():
    # Config name for Sports and Outdoors reviews
    # config_name = "raw_review_Sports_and_Outdoors"
    # print(f"Loading dataset: {config_name} ...")
    # dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", config_name, trust_remote_code=True)
    # ds = dataset["full"]
    # print(f"Loaded {len(ds)} records for {config_name}.")

    # # Output JSONL file
    # output_file = "raw_review_Sports_and_Outdoors.jsonl"
    # parallel_write_jsonl(ds, output_file, num_workers=mp.cpu_count(), chunk_size=20000)

    # Also download the metadata for Sports and Outdoors
    metadata_config_name = "raw_meta_Sports_and_Outdoors"
    print(f"Loading metadata dataset: {metadata_config_name} ...")
    metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", metadata_config_name, trust_remote_code=True)
    metadata_ds = metadata_dataset["full"]
    print(f"Loaded {len(metadata_ds)} records for {metadata_config_name}.")

    # Output JSONL file for metadata
    metadata_output_file = "metadata_Sports_and_Outdoors.jsonl"
    parallel_write_jsonl(metadata_ds, metadata_output_file, num_workers=mp.cpu_count(), chunk_size=20000)

if __name__ == "__main__":
    main()