#!/bin/bash
#SBATCH --job-name=rec
#SBATCH --output=logs/%x-%j.out         # stdout
#SBATCH --error=logs/%x-%j.err          # stderr
#SBATCH --partition=P2
#SBATCH --nodes=1                  # single node
#SBATCH --cpus-per-task=8         # CPU cores per task
#SBATCH --gres=gpu:2               # 1 GPUs
#SBATCH --mem=100G                 # memory
#SBATCH --exclude=b[19,31]

# (Optional) Load modules, e.g., if your cluster uses them
# module load cuda/11.7
# module load anaconda/2023a

# Source your .bashrc which has the Conda init lines
# (Make sure .bashrc has the 'conda init' lines)

export N_GPUS=2
export DATA_DIR=data/amazon_c4/inst/dense/sports/512
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=matching-qwen2.5-3b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="c3d83ce1111ba40831b20b693213bdba81a52688" # 여기 본인 Wandb api KEY입력
export HF_HOME="/home/s1/sehopyo/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1

conda init

source ~/.bashrc

conda activate zero

# Head 노드 시작 (현재 Slurm이 할당한 노드에서)
head_ip=$(hostname --ip-address)
ray start --head --port=6379 --node-ip-address=$head_ip --memory=20000000000 --plasma-directory=/tmp


DATE=$(date '+%Y-%m-%d-%H-%M-%S')

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 2>&1 | tee exp_log/$EXPERIMENT_NAME-grpo-verl_2gpus_$DATE.log