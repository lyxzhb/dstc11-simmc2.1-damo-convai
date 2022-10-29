cd ..

data_dir=data

log_dir=log
time_dir=$(date "+%m%d%H%M")
mkdir ./$log_dir/task3_$time_dir

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup /opt/conda/envs/simmc/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port 10014 run_dstc11_task3_ddp.py \
  --item2id=../$data_dir/item2id.json \
  --train_input_file=../$data_dir/simmc2.1_dials_dstc11_task3_predict.json \
  --eval_input_file=../$data_dir/simmc2.1_dials_dstc11_task3_eval.json \
  --furniture_slot_map_file=../$data_dir/furniture_slot_map.json \
  --fashion_slot_map_file=../$data_dir/fashion_slot_map.json \
  --add_special_tokens=../$data_dir/simmc2_special_tokens.json \
  --output_dir=../save_model \
  --backbone=allenai/longformer-base-4096 \
  --train_batch_size=16 \
  --eval_batch_size=4 \
  --learning_rate=5e-5 \
  --warmup_rate=0.5 \
  --num_workers=96 \
  --num_train_epochs=300 > ./$log_dir/task3_$time_dir/config.txt 2>./$log_dir/$time_dir/output.txt 


# Evaluate
CUDA_VISIBLE_DEVICES=0 python eval_vlbert_multi_task_task3.py \
  --item2id=./$data_dir/item2id.json \
  --train_input_file=./$data_dir/simmc2.1_dials_dstc11_task3_predict.json \
  --eval_input_file=./$data_dir/simmc2.1_dials_dstc11_task3_eval.json \
  --furniture_slot_map_file=./$data_dir/furniture_slot_map.json \
  --fashion_slot_map_file=./$data_dir/fashion_slot_map.json \
  --add_special_tokens=./$data_dir/simmc2_special_dst_tokens.json \
  --output_dir=./save_model/dstc11-checkpoint \
  --backbone=allenai/longformer-base-4096 \
  --checkpoint_name_or_path=allenai/longformer-base-4096 \
  \
  --train_batch_size=12 \
  --eval_batch_size=4 \
  --learning_rate=5e-5 \
  --warmup_rate=0.4 \
  --num_workers=96 \
  --num_train_epochs=200