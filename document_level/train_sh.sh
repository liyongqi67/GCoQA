#!/usr/bin/env bash




python3  train_query_encoder.py \
      --do_train True \
      --do_gen True \
      --do_test True \
      --load_small True \
      --num_train_epochs 1 \
      --per_gpu_train_batch_size 4 \
      --per_gpu_eval_batch_size 4 \
      --per_gpu_test_batch_size 2 \
      --overwrite_output_dir True \