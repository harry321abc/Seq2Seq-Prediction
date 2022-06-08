### M
export CUDA_VISIBLE_DEVICES=1
cd ..
python -u main_informer.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id check \
  --model transformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --des 'Exp' \
  --itr 1