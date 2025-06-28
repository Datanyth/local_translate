export CUDA_VISIBLE_DEVICES=0
python -m src.main \
  --model_id LLaMAX/LLaMAX3-8B-Alpaca \
  --repo_id ChaosAiVision/Deepseek_R1_vi \
  --subset train \
  --src_language English \
  --trg_language Vietnamese \
  --max_length_token 12800 \
  --dataset_name knoveleng/open-s1 \
  --column_name problem solution \
  --translated_dataset_dir ".cache" \
  --download_dataset_dir ".cache" \
  --start_inter 0 \
  --writer_batch_size 20 \
  --use_4bit
