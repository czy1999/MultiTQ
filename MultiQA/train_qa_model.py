import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np
from qa_baselines import QA_baseline, QA_lm, QA_embedkgqa, QA_cronkgqa, QA_MultiQA
from qa_datasets import QA_Dataset_cron, QA_Dataset_multi, QA_Dataset_Baseline_cron, QA_Dataset_Baseline_muti, QA_Dataset_MultiQA_muti
from torch.utils.data import Dataset, DataLoader
import utils
from tqdm import tqdm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info, save_model, append_log_to_file, train, eval_cron,eval_multi
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='tcomplex.ckpt', type=str,
    help="Pretrained tkbc model checkpoint enhanced_icews.ckpt"
)

parser.add_argument(
    '--dataset_name', default='MultiTQ', type=str,
    help="Which dataset to use."
)

parser.add_argument(
    '--sub_dataset', default='processed_questions', type=str,
    help="Which sub dataset to use. only for MultiTQ dataset."
)

parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)
parser.add_argument(
    '--kg_dir', default='kg', type=str,
    help="Which kg directory to use"
)
parser.add_argument(
    '--model', default='cronkgqa', type=str,
    help="Which model to use."
)

parser.add_argument(
    '--lm_model', default='distilbert-base-uncased', type=str,
    help="Which language model to use."
)

parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)

parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=10, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)

parser.add_argument(
    '--valid_freq', default=1, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--batch_size', default=100, type=int,
    help="Batch size."
)

parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)

parser.add_argument(
    '--frozen', default=1, type=int,
    help="Whether entity/time embeddings are frozen or not. Default frozen."
)

parser.add_argument(
    '--lm_frozen', default=1, type=int,
    help="Whether language model params are frozen or not. Default frozen."
)

parser.add_argument(
    '--lr', default=2e-4, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)

parser.add_argument(
    '--eval_split', default='dev', type=str,
    help="Which split to validate on"
)

parser.add_argument(
    '--extra_entities', default=False, type=bool,
    help="For some question types."
)

parser.add_argument(
    '--test', default="test", type=str,
    help="Test data."
)

parser.add_argument(
    '--max_questions', default=50000, type=int,
    help="Test data."
)

args = parser.parse_args()
print_info(args)

if args.dataset_name == 'CronQuestions':
    QA_Dataset_Baseline = QA_Dataset_Baseline_cron
    eval = eval_cron
elif args.dataset_name == 'MultiTQ':
    QA_Dataset_Baseline = QA_Dataset_Baseline_muti
    QA_Dataset_MultiQA = QA_Dataset_MultiQA_muti
    eval = eval_multi
else:
    print('Unknown dataset name')
    exit(1)

if args.model != 'embedkgqa':
    tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))
else:
    tkbc_model = loadTkbcModel_complex('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
        dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    ))

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.eval_split)
    exit(0)

train_split = 'train'
test = 'test'
if args.model == 'bert' or args.model == 'roberta':
    qa_model = QA_lm(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name, args=args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name, args=args)

elif args.model == 'embedkgqa':
    qa_model = QA_embedkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name, args=args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name, args=args)

elif args.model == 'cronkgqa':
    qa_model = QA_cronkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name, args=args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split, dataset_name=args.dataset_name , args=args)
    test_dataset = QA_Dataset_Baseline(split=test, dataset_name=args.dataset_name, args=args)

elif args.model == 'multiqa':
    dataset = QA_Dataset_MultiQA(split=train_split, dataset_name=args.dataset_name, args=args)
    valid_dataset = QA_Dataset_MultiQA(split=args.eval_split,dataset_name=args.dataset_name,args = args)
    test_dataset = QA_Dataset_MultiQA(split=test, dataset_name=args.dataset_name, args=args)
    qa_model = QA_MultiQA(tkbc_model, args)
else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)

if args.load_from != '':
    filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.load_from
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

if args.mode == 'eval':
    score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split=args.eval_split, k=args.eval_k)
    exit(0)

result_filename = 'results/{dataset_name}/{model_file}.log'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)

train(qa_model, dataset, valid_dataset, args, result_filename=result_filename)


filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
    dataset_name=args.dataset_name,
    model_file=args.save_to
)
print('Loading best model from', filename)
qa_model.load_state_dict(torch.load(filename))
score, log = eval(qa_model, test_dataset, batch_size=args.valid_batch_size, split="test", k=args.eval_k)
log = ["######## TEST EVALUATION FINAL (BEST) #########"] + log
append_log_to_file(log, 0, result_filename)

print('Training finished')
