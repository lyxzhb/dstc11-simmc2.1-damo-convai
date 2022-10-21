cd ..

data_dir=data_dstc11

log_dir=log
time_dir=task1_$(date "+%m%d%H%M")
mkdir ./$log_dir/$time_dir

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /opt/conda/envs/simmc/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port 10020 run.py \
#   --item2id=$data_dir/item2id.json \
#   --train_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_predict.json \
#   --eval_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_eval.json \
#   --add_special_tokens=$data_dir/simmc2_special_tokens.json \
#   --output_dir=./save_model/dstc11-checkpoint \
#   --backbone=allenai/longformer-base-4096 \
#   --train_batch_size=16 \
#   --eval_batch_size=4 \
#   --learning_rate=5e-5 \
#   --warmup_rate=0.5 \
#   --num_workers=96 \
#   --num_train_epochs=60 > ./$log_dir/$time_dir/config.txt 2>./$log_dir/$time_dir/output.txt 


CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/simmc/bin/python -m torch.distributed.launch --nproc_per_node=1 --master_port 10020 run_dstc11_task1_ddp_with_disam.py \
  --item2id=$data_dir/item2id.json \
  --train_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_phrase2_predict.json \
  --eval_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_phrase2_eval.json \
  --add_special_tokens=$data_dir/simmc2_special_tokens.json \
  --output_dir=./save_model/dstc11-checkpoint \
  --backbone=./save_model/backbone/longformer-base-4096 \
  --train_batch_size=12 \
  --eval_batch_size=4 \
  --learning_rate=5e-5 \
  --warmup_rate=0.1 \
  --num_workers=96 \
  --num_train_epochs=60 > ./$log_dir/$time_dir/config.txt 2>./$log_dir/$time_dir/output.txt 


# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10011 run_dstc11_task1_ddp_with_disam.py \
#   --item2id=$data_dir/item2id.json \
#   --train_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_phrase2_predict.json \
#   --eval_input_file=$data_dir/task1/simmc2.1_dials_dstc11_task1_phrase2_eval.json \
#   --add_special_tokens=$data_dir/simmc2_special_tokens.json \
#   --output_dir=./save_model/dstc11-checkpoint \
#   --backbone=./save_model/backbone/longformer-base-4096 \
#   --train_batch_size=12 \
#   --eval_batch_size=4 \
#   --learning_rate=5e-5 \
#   --warmup_rate=0.5 \
#   --num_workers=96 \
#   --num_train_epochs=60
