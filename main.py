import os
import ipdb
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import argparse
import logging

import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup 

from processor import WSD_Processor
from model import BertForWSD
from utils import set_seed, get_pred_label


logger = logging.getLogger(__name__)


def train(args, model, processor):
    set_seed(args)

    logger.info("train dataloader generation")
    _, train_features, train_dataloader = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    _, dev_features, dev_dataloader = processor.generate_dataloader('dev')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader)*args.train_batch_size)
    logger.info("  train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    smooth_loss = 0.0
    best_dev_acc = 0.0

    model.zero_grad()
    while global_step <= args.max_steps:
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {'sent_token_ids':       batch[0].to(args.device),
                'sent_mask_ids':  batch[1].to(args.device), 
                'start_locs': batch[2].to(args.device), 
                'end_locs': batch[3].to(args.device), 
                'labels': batch[4].to(args.device), 
                'sense_token_ids_list': [item.to(args.device) for item in batch[5]],
                'sense_mask_ids_list':  [item.to(args.device) for item in batch[6]], 
            }         
            loss, _ = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            smooth_loss += loss.item()/args.logging_steps
            if (step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if global_step % args.logging_steps == 0:
                logging.info("-----------------------global_step: {} -------------------------------- ".format(global_step))
                logging.info('lr: {}'.format(scheduler.get_lr()[0]))
                logging.info('smooth_loss: {}'.format(smooth_loss))
                smooth_loss = .0

            if global_step % args.eval_steps == 0:
                dev_acc = evaluate(args, model, dev_features, dev_dataloader)
                
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    model.save()
            
            
def evaluate(args, model, features, dataloader):
    gt_list = []
    pred_list = []
    for batch in dataloader:
        model.eval()
        with torch.no_grad(): 
            inputs = {'sent_token_ids':       batch[0].to(args.device),
                'sent_mask_ids':  batch[1].to(args.device), 
                'start_locs': batch[2].to(args.device), 
                'end_locs': batch[3].to(args.device), 
                'sense_token_ids_list': [item.to(args.device) for item in batch[5]],
                'sense_mask_ids_list':  [item.to(args.device) for item in batch[6]], 
            }     
            score_list = model(**inputs)
            
        gt = batch[4].cpu().tolist()
        pred = get_pred_label(score_list)
        gt_list.extend(gt)
        pred_list.extend(pred)
    
    correct_num = sum([(gt==pred) for (gt, pred) in zip(gt_list, pred_list)])
    acc = correct_num/len(gt_list)
    logger.info("Acc:{}\tgt num:{}\tcorrect num:{}".format(acc, len(gt_list), correct_num))
    
    assert(len(features)==len(pred_list))
    with open(os.path.join(args.output_dir, 'pred.txt'), 'w', encoding='utf-8') as f:
        for (feature, pred) in zip(features, pred_list):
            idx = feature.idx
            sense = feature.sense_name_list[pred]
            f.write('{} {}\n'.format(idx, sense))
    
    return acc
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./WSD_Evaluation_Framework/Training_Corpora/SemCor")
    parser.add_argument("--dev_file", default="./WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007")
    parser.add_argument("--test_file", default="./WSD_Evaluation_Framework/Evaluation_Datasets/ALL")
    
    parser.add_argument("--train_dataset_name", default="semcor")
    parser.add_argument("--dev_dataset_name", default="semeval2007")
    parser.add_argument("--test_dataset_name", default="ALL")
    
    parser.add_argument("--wn_path", default="./WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt")
    parser.add_argument("--output_dir", default='./outputs_exp', type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    
    parser.add_argument("--pad_mask_token", default=0, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--max_text_length", default=192, type=int)
    parser.add_argument("--max_gloss_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--infer_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--max_steps", default=200000, type=int)
    parser.add_argument("--warmup_steps", default=0.05, type=float)
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--eval_steps', default=1000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--loss_self', default=False, action='store_true')
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--inference_only", default=False, action="store_true")
    parser.add_argument("--load_checkpoints", default="best_model.ckpt", type=str)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.inference_only:      
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
    else:
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
    set_seed(args)

    config = BertConfig.from_pretrained(args.model_name_or_path)
    config.output_dir = args.output_dir
    config.model_name_or_path = args.model_name_or_path
    config.loss_self = args.loss_self
    config.normalize = args.normalize
    config.device = args.device

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, add_special_tokens=True)
    model = BertForWSD(config=config)
    model.to(args.device)

    processor = WSD_Processor(args, tokenizer)
    
    if not args.inference_only:
        logger.info("Training/evaluation parameters %s", args)
        train(args, model, processor)
    else:
        logger.info("dev dataloader generation")
        _, dev_features, dev_dataloader = processor.generate_dataloader('dev')
        model.load(args.load_checkpoints)
        evaluate(args, model, dev_features, dev_dataloader)
            

if __name__ == "__main__":
    main()