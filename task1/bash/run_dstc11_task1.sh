cd ..

data_dir=data

log_dir=log
time_dir=task1_$(date "+%m%d%H%M")
mkdir ./$log_dir/$time_dir


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port 10020 run_dstc11_task1_ddp.py \
  --item2id=../$data_dir/item2id.json \
  --train_input_file=../$data_dir/simmc2.1_dials_dstc11_task1_phrase2_predict.json \
  --eval_input_file=../$data_dir/simmc2.1_dials_dstc11_task1_phrase2_eval.json \
  --add_special_tokens=../$data_dir/simmc2_special_tokens.json \
  --output_dir=../save_model \
  --backbone=allenai/longformer-base-4096 \
  --train_batch_size=64 \
  --eval_batch_size=4 \
  --learning_rate=5e-5 \
  --warmup_rate=0.1 \
  --num_workers=128 \
  --num_train_epochs=300 > ../$log_dir/$time_dir/config.txt 2>../$log_dir/$time_dir/output.txt 

