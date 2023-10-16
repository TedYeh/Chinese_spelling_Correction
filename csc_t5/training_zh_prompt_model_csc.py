import argparse
import json
from loguru import logger
import pandas as pd
import numpy as np
import time
import os
import sys
import random
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
sys.path.append('../..')
from t5.t5_model import T5Model
from t5.copyt5_model import CopyT5Model
from t5.t5_utils import f1_sim, rouge_l_zh
from opencc import OpenCC


def load_json_data(prefix, file_path, data_type='train'):
    data = []
    n = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                json_string = json.loads(line.strip())
                if data_type == 'train':
                    input_text = json_string["original_text"] + "_輸出句："
                    target_text = json_string["correct_text"]
                    '''
                    if np.random.choice([True, False], p=[0.5, 0.5]):# and n < 3e4:
                        input_text = '糾正句子中的錯字：' + json_string["original_text"] + "_輸出句："
                        target_text = json_string["correct_text"]
                        n += 1
                    else: 
                        input_text = '找到句子中的錯字：' + json_string["original_text"] + "_錯別字："
                        target_text = json_string["wrong_chr"]
                    '''
                else:
                    input_text = '糾正句子中的錯字：' + json_string["original_text"] + "_輸出句："
                    #input_text = json_string["original_text"]
                    target_text = json_string["correct_text"]
                #answer_choices = json_string.get("answer_choices", [])
                #type = json_string["type"]
                data.append([prefix, input_text, target_text])
            else:
                logger.warning(f'line error: {line}')
        print(n)
    return data
cc = OpenCC('s2twp')  #簡體中文 -> 繁體中文
def preprocess_function(example):
    global cc
    original_text, wrong_ids, correct_text = example["original_text"], example["wrong_ids"], example["correct_text"]
    #example['instruction'] = '对下面中文拼写纠错：'    
    example["prefix"] = '糾正句子中的錯字：'
    example['input_text'] = '糾正句子中的錯字：' + cc.convert(original_text) 
    example['target_text'] = cc.convert(correct_text) 
    return example


def load_data(data):
    if data.endswith('.json') or data.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=data)
    elif os.path.isdir(data):
        dataset = load_from_disk(data)
    else:
        dataset = load_dataset(data)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(preprocess_function, batched=False)
    eval_dataset = dataset["validation"]
    eval_dataset = eval_dataset.map(preprocess_function, batched=False)
    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(preprocess_function, batched=False)#, remove_columns=test_dataset.column_names
    return train_dataset.to_pandas(), eval_dataset.to_pandas(), test_dataset.to_pandas()

def normalize(text):
    """文本標準化"""
    return ' '.join(text.lower().split())

def main(is_simple = False):
    parser = argparse.ArgumentParser() #ClueAI/PromptCLUE-base-v1-5 #IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese
    parser.add_argument('--train_file', default='./data/cgedit.json', type=str, help='Training data file')
    parser.add_argument('--test_file', default='./data/csc_test.json', type=str, help='Test data file')
    parser.add_argument('--model_type', default='t5', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='ClueAI/PromptCLUE-base-v1-5', type=str, help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--prefix', default='prompt', type=str, help='Prefix str')
    parser.add_argument('--output_dir', default='./outputs/prompt_cgedit', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=200, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=512, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    torch.cuda.set_device(0)
    if args.do_train:
        logger.info('Loading data...')
        if not is_simple:
            train_data = load_json_data(args.prefix, args.train_file, 'train')
            logger.debug('train_data: {}'.format(train_data[:10]))
            train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])
            train_df, eval_df = train_test_split(train_df, test_size=0.05, random_state=2023)
        else:
            train_df, eval_df, test_df = load_data('shibing624/CSC')
            logger.debug('train_data: {}'.format(train_df['input_text']))
            #input()
        #eval_data = load_json_data(args.prefix, args.train_file)[:10]
        #eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": True,
            "evaluate_generated_text":True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "save_best_model": True,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
            "best_model_dir": os.path.join(args.output_dir, "best_model"),
        }
        model = T5Model(args.model_type, args.model_name, args=model_args)
        #model = CopyT5Model(args.model_type, args.model_name, args=model_args)
        def sim_text_chars(text1, text2):
            if not text1 or not text2:
                return 0.0
            same = set(text1) & set(text2)
            m = len(same)
            n = len(set(text1)) if len(set(text1)) > len(set(text2)) else len(set(text2))
            return m / n

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([sim_text_chars(label, pred) for label, pred in zip(labels, preds)]) / len(labels)
            logger.debug(f"match: {match}")
            return match
        
        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        model = T5Model(args.model_type, args.output_dir, args={"eval_batch_size": args.batch_size}, evaluate=True)
        if not is_simple:
            test_data = load_json_data(args.prefix, args.test_file, 'test')
            test_df = pd.DataFrame(test_data, columns=["prefix", "input_text", "target_text"])
        logger.debug('test_df: {}'.format(test_df))        
        to_predict = [
                    input_text
                    for prefix, input_text in zip(
                        test_df["prefix"], test_df["input_text"]
                    )
                ]
        test_df['predict_after'] = model.predict(to_predict)
        out_df = test_df[["input_text", "target_text", 'predict_after']]
        out_df.to_json('test_result.json', force_ascii=False, orient='records', lines=True)
        


if __name__ == '__main__':
    main(False)
