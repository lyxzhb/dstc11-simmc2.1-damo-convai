from genericpath import exists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from os.path import join
import json
import argparse

import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler, DistributedSampler
from tqdm import tqdm, trange
from datetime import datetime
import torch.multiprocessing
from sklearn.metrics import precision_recall_fscore_support
from rich import print
torch.multiprocessing.set_sharing_strategy('file_system')
torch.distributed.init_process_group(backend="nccl")

from transformers import (
    LongformerTokenizerFast,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from torch import distributed as dist

from utils.metadata import load_metadata,available_sizes2st
from utils.set_config import set_device, set_seed_ddp

from model.backbone import VLBertModelWithDST
from utils.dataset import get_dst_dataset, DataLoaderX


def train(args, model, tokenizer, all_objects_meta, fashion_slot_map, furniture_slot_map):

    def collate_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        boxes = list(map(lambda x: x[2], examples))
        misc = list(map(lambda x: x[3], examples))
        nocoref = list(map(lambda x: x[4], examples))
        disambiguation_labels = list(map(lambda x: x[5], examples))
        intent = list(map(lambda x: x[6], examples))
        slot_values = list(map(lambda x: x[7], examples))
        
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)

        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)

        return  enc_input_pad, \
                enc_attention_pad, \
                boxes, \
                misc, \
                nocoref, \
                torch.vstack(disambiguation_labels), \
                intent, \
                slot_values

    train_dataset = get_dst_dataset(args, tokenizer, all_objects_meta, eval=False, fashion_slot_map=fashion_slot_map, furniture_slot_map=furniture_slot_map)
    # train_sampler = RandomSampler(train_dataset)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoaderX(train_dataset, num_workers=args.num_workers, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_bart, pin_memory=True)
    t_total = len(train_dataloader) * args.num_train_epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_rate * t_total,
        num_training_steps=t_total
    )

    global_step = 0
    best_f1_score = 0.0
    save_checkpoints = []

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(  model, 
                                                        device_ids=[args.local_rank], 
                                                        output_device=args.local_rank, 
                                                        find_unused_parameters=True, 
                                                        broadcast_buffers=False) 
    

    model.zero_grad()
        
    for epoch_idx in range(args.num_train_epochs):

        train_sampler.set_epoch(epoch_idx)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], colour='green', leave=False)
        model.train()

        for batch_idx, batch in enumerate(epoch_iterator):

            # 提取batch data
            enc_input = batch[0].to(args.device)
            enc_attention_mask = batch[1].to(args.device)
            boxes = batch[2]  # batch, num_obj_per_line, 6
            misc = batch[3]  # batch, num_obj_per_line, dict  # 这个misc是什么信息
            nocoref = batch[4]
            disambiguation_labels = batch[5].to(args.device)
            intent = batch[6]
            slot_values = batch[7]
                
            step_loss = model(enc_input, enc_attention_mask, boxes, misc, nocoref, disambiguation_labels, intent, slot_values)
            global_step += 1

            epoch_iterator.set_postfix(epoch=epoch_idx, global_step=global_step, step_loss=step_loss.item())

            optimizer.zero_grad()
            step_loss.backward()
            
            parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            # if global_step % args.embedding_train_steps == 0:  # 如果全局步骤在embedding_train_step取余为0时在训练一次embedding
            #     train_embedding_clip_way(args, model, tokenizer, all_objects_meta, args.embedding_train_epochs_ongoing, do_tsne=False)

        if epoch_idx >= 3 and args.local_rank in [-1, 0] and (epoch_idx % 2 == 1 or epoch_idx == args.num_train_epochs - 1):
        # if args.local_rank in [-1, 0]:
            # Evaluation
            model.eval()
            total_report = evaluate(args, model, tokenizer, all_objects_meta, fashion_slot_map, furniture_slot_map)
            total_report['epoch_idx'] = epoch_idx
            print('EVALUATION:', total_report)

            if total_report['slot_value_f1-score'] > best_f1_score:
                # Save checkpoint
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, epoch_idx))
                save_checkpoints.append(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                torch.save(model.module.state_dict(), os.path.join(output_dir, "model.bin"))
                
                with open(join(output_dir, 'report.json'), 'w') as f_out:
                    json.dump(total_report, f_out, indent=4, ensure_ascii=False)
                best_f1_score = total_report['slot_value_f1-score']

                if len(save_checkpoints) > args.save_checkpoints:
                    # 删除保存的checkpoints
                    try:
                        rm_dir = save_checkpoints.pop(0)
                        for file in os.listdir(rm_dir):
                            os.remove(join(rm_dir, file))
                        os.rmdir(rm_dir)
                    except:
                        print('DELETE Checkpoints Error')
                    
        dist.barrier() 

    return global_step


def evaluate(args, model, tokenizer, all_objects_meta, fashion_slot_map, furniture_slot_map):
    ''' 模型方法的评估函数'''
    def collate_eval_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        boxes = list(map(lambda x: x[2], examples))
        misc = list(map(lambda x: x[3], examples))
        disambiguation_labels = list(map(lambda x: x[5], examples))
        intent = list(map(lambda x: x[6], examples))
        slot_values = list(map(lambda x: x[7], examples))
        
        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)

        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)

        return enc_input_pad, enc_attention_pad, boxes, misc, torch.vstack(disambiguation_labels).squeeze(), intent, slot_values

    def rec_prec_f1(n_correct, n_true, n_pred):
        rec = n_correct / n_true if n_true != 0 else 0
        prec = n_correct / n_pred if n_pred != 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0
        return rec, prec, f1

    eval_dataset = get_dst_dataset(args, tokenizer, all_objects_meta, eval=True, fashion_slot_map=fashion_slot_map, furniture_slot_map=furniture_slot_map)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoaderX(eval_dataset, sampler=eval_sampler, num_workers=args.num_workers, batch_size=args.eval_batch_size, collate_fn=collate_eval_bart, pin_memory=True)

    n_pred_objects, n_true_objects, n_correct_objects = 0, 0, 0
    n_total_disambiguation, n_true_disambiguation = 0, 0
    intent_target_list, intent_pred_list = [], []
    slot_values_target_list, slot_values_pred_list = [], []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", colour='blue', leave=False):
        enc_input = batch[0].to(args.device)
        enc_attention_mask = batch[1].to(args.device)
        boxes = batch[2]  # batch, num_obj_per_line, 6
        misc = batch[3]  # batch, num_obj_per_line, dict
        disambiguation_labels = batch[4].to(args.device)
        intent = batch[5]
        slot_values = batch[6]
        
        with torch.no_grad():
            s_pred_objects, s_true_objects, s_correct_objects, disambiguation_true_items, disambiguation_total_items, intent_target, intent_pred, slot_values_target, slot_values_pred = model.module.evaluate(enc_input, enc_attention_mask, boxes, misc, disambiguation_labels, intent, slot_values)
            n_pred_objects += s_pred_objects
            n_true_objects += s_true_objects
            n_correct_objects += s_correct_objects
            n_true_disambiguation += disambiguation_true_items
            n_total_disambiguation += disambiguation_total_items

            intent_target_list.extend(intent_target)
            intent_pred_list.extend(intent_pred)
            slot_values_target_list.extend(slot_values_target)
            slot_values_pred_list.extend(slot_values_pred)
    
    intent_pre, intent_rec, intent_f1, sup = precision_recall_fscore_support(intent_target_list, intent_pred_list)
    
    n_correct_slot_values, n_true_slot_values, n_pred_slot_values = 0, 0, 0
    
    for idx in range(len(slot_values_target_list)):
        for key in slot_values_target_list[idx].keys():
            if key == 'availableSizes':
                if slot_values_target_list[idx][key] != [0, 0, 0, 0, 0, 0]:
                    n_true_slot_values += 1
                if slot_values_pred_list[idx][key] != [0, 0, 0, 0, 0, 0]:
                    n_pred_slot_values += 1
                if slot_values_target_list[idx][key] == slot_values_pred_list[idx][key] and slot_values_pred_list[idx][key] != [0, 0, 0, 0, 0, 0]:
                    n_correct_slot_values += 1
            else:
                if slot_values_target_list[idx][key] != 0:
                    n_true_slot_values += 1
                if slot_values_pred_list[idx][key] != 0:
                    n_pred_slot_values += 1
                if slot_values_target_list[idx][key] == slot_values_pred_list[idx][key] and slot_values_pred_list[idx][key] != 0:
                    n_correct_slot_values += 1
    
    coref_rec, coref_prec, coref_f1 = rec_prec_f1(n_correct_objects, n_true_objects, n_pred_objects)
    slot_values_rec, slot_values_prec, slot_values_f1 = rec_prec_f1(n_correct_slot_values, n_true_slot_values, n_pred_slot_values)
    
    return {
        'precision': coref_prec,
        'recall': coref_rec,
        'f1-score': coref_f1,
        'disambiguation_acc': n_true_disambiguation/n_total_disambiguation,
        'slot_value_precision': slot_values_prec,
        'slot_value_recall': slot_values_rec,
        'slot_value_f1-score': slot_values_f1,
        'intent_precision': intent_pre.mean(),
        'intent_recall': intent_rec.mean(),
        'intent_f1-score': intent_f1.mean(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_batch_size',
        default=4,
        type=int,
    )
    parser.add_argument(
        '--eval_batch_size',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--num_train_epochs',
        default=3,
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        default=64,
        type=int,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_rate",
        default=0.1,
        type=float, 
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--embedding_train_steps",
        default=200,
        type=int
    )
    parser.add_argument(
        "--save_checkpoints",
        default=3,
        type=int
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--item2id",
        required=True,
        type=str,
        help='item2id filepath'
    )
    parser.add_argument(
       "--train_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
       "--fashion_slot_map_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
       "--furniture_slot_map_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--backbone",
        default="bert-large-uncased",
        type=str,
        help='backbone of model'
    )
    parser.add_argument(
        "--eval_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    args = parser.parse_args()

    if args.output_dir:  # 获取 TimeString Dir
        args.output_dir = join(args.output_dir, datetime.now().strftime("%m%d%H%M")+'_vlbert_task3')
        if not exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank in [-1, 0]:
        with open(join(args.output_dir, 'config.json'), 'w') as f_in:
            json.dump(vars(args), f_in, indent=4, ensure_ascii=False)

    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed_ddp(args)
    set_device(args)

    tokenizer = LongformerTokenizerFast.from_pretrained(args.backbone)
    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        args.len_tokenizer = len(tokenizer)
    
    # Define Model
    model = VLBertModelWithDST(args)
    model.to(args.device)

    # meta的信息转化为token id：<@1000>起/<@2000>起
    with open(args.item2id, 'r') as f:
        item2id = json.load(f)

    with open(args.fashion_slot_map_file, 'r') as f_in:
        fashion_slot_map = json.load(f_in)
    with open(args.furniture_slot_map_file, 'r') as f_in:
        furniture_slot_map = json.load(f_in)
        
    fashion_meta, furniture_meta = load_metadata('/'.join(args.item2id.split('/')[:-1]))

    all_objects_meta = dict()

    for meta in fashion_meta:
        object_special_id = item2id[meta.name]
        object_meta = {
            'asset_type': meta.asset_type,
            'customer_review': str(meta.customer_review),
            'available_sizes': [available_sizes2st[size] for size in meta.available_sizes],
            'color': meta.color,
            'pattern': meta.pattern,
            'brand': meta.brand,
            'sleeve_length': meta.sleeve_length,
            'type': meta.type,
            'price': str(meta.price),
            'size': meta.size
        }  # Fashion领域有10项Feature (Visual/Not Visual)
        all_objects_meta[object_special_id] = object_meta

    for meta in furniture_meta:
        object_special_id = item2id[meta.name]
        object_meta = {
            'brand': meta.brand,
            'color': meta.color,
            'customer_review': str(meta.customer_review),
            'materials': meta.materials,
            'price': meta.price,
            'type': meta.type
        }  # Furniture领域有6项Feature (Visual/Not Visual)
        all_objects_meta[object_special_id] = object_meta
    
    if args.local_rank in [-1, 0]:
        print()
        print(vars(args))
        print()

    # 展开训练
    global_step = train(args, model, tokenizer, all_objects_meta, fashion_slot_map, furniture_slot_map)


if __name__ == '__main__':
    main()
