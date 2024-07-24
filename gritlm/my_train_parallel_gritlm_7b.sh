conda activate grit
export WANDB_PROJECT="embedding"

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 1 \
  -m training.run \
  --output_dir /data/lora-models/test_grit_2 \
  --model_name_or_path /data/models/GritLM-7B \
  --train_data /root/code/gritlm/gritlm/training/toy_data/toy_data_embedding.jsonl \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --dataloader_drop_last True \
  --normalized True \
  --temperature 0.02 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --train_group_size 2 \
  --mode embedding \
  --bf16 \
  --attn_implementation sdpa \
  --no_gen_gas \
  --no_emb_gas \
  --split_emb \
  --attn cccc
