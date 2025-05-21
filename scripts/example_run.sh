export CUDA_VISIBLE_DEVICES=2
python -m src.main \
  --model_id LLaMAX/LLaMAX3-8B-Alpaca \
  --repo_id ChaosAiVision/Deepseek_R1_vi \
  --subset train \
  --src_language English \
  --trg_language Vietnamese \
  --max_length_token 12800 \
  --dataset_name knoveleng/open-s1 \
  --column_name problem solution \
  --translated_dataset_dir /home/datnvt/chaos/code/trash/data_translated \
  --download_dataset_dir /home/datnvt/chaos/code/trash/save_data_hf \
  --start_inter 0 \
  --writer_batch_size 20
