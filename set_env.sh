export N_GPUS=2
export DATA_DIR=data/amazon_c4/sports_parquet_review_added
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=matching-qwen2.5-3b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="9a4a2f1180aaab5279697e1dd3bc3ef129b76422" # 여기 본인 Wandb api KEY입력
export HF_HOME="$HOME/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0,1

