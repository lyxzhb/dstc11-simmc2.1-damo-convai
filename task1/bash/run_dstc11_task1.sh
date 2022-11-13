cd ..

data_dir=data

log_dir=log
time_dir=task1_$(date "+%m%d%H%M")
mkdir ./$log_dir/$time_dir

# Evaluate
CUDA_VISIBLE_DEVICES=0 python eval_dstc11_task1.py \
  --item2id=./$data_dir/item2id.json \
  --train_input_file=./$data_dir/simmc2.1_dials_dstc11_task1_phrase2_predict.json \
  --eval_input_file=./$data_dir/simmc2.1_dials_dstc11_task1_phrase2_eval_teststd.json \
  --add_special_tokens=./$data_dir/simmc2_special_tokens.json \
  --output_dir=./save_model \
  --backbone=allenai/longformer-base-4096 \
  --eval_disam_path=<YOUR MODEL TASK2 CHECKPOINTS> \
  --checkpoint_name_or_path=<YOUR MODEL TASK1 CHECKPOINTS> \
  --train_batch_size=12 \
  --eval_batch_size=4 \
  --learning_rate=5e-5 \
  --warmup_rate=0.4 \
  --num_workers=96 \
  --num_train_epochs=200