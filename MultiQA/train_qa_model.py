import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np

from qa_baselines import QA_cronkgqa_RT, QA_timeQA2, QA_cronkgqa_soft, QA_baseline, QA_lm, QA_embedkgqa, QA_cronkgqa, \
    QA_MG_transformer, QA_MG_selected_transformer, QA_multiqa

from qa_datasets import QA_Dataset, QA_Dataset_Baseline, QA_Dataset_MG

from torch.utils.data import Dataset, DataLoader
import utils
from tqdm import tqdm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info, save_model, append_log_to_file,train, eval
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description="MulitiQA KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='tkbc_tcomplex_256.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)
parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)

parser.add_argument(
    '--model', default='MultiQA', type=str,
    help="Which model to use."
)
parser.add_argument(
    '--supervision', default='soft', type=str,
    help="Which supervision to use."
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
    '--max_epochs', default=20, type=int,
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
    '--batch_size', default=128, type=int,
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
    '--dataset_name', default='processed_questions', type=str,
    help="Which dataset."
)
parser.add_argument(
    '--lm', default='distilbert', type=str,
    help="Lm to use."
)
parser.add_argument(
    '--fuse', default='add', type=str,
    help="For fusing time embeddings."
)
parser.add_argument(
    '--extra_entities', default=False, type=bool,
    help="For some question types."
)
parser.add_argument(
    '--corrupt_hard', default=0., type=float,
    help="For some question types."
)

parser.add_argument(
    '--test', default="test", type=str,
    help="Test data."
)

args = parser.parse_args()
print_info(args)

if args.model != 'embedkgqa':  # TODO this is a hack
    # tkbc_model = loadTkbcModel('models/{dataset_name}/kg_embeddings/{tkbc_model_file}'.format(
    #     dataset_name=args.dataset_name, tkbc_model_file=args.tkbc_model_file
    # ))
    tkbc_model = loadTkbcModel('models/kg_embeddings/{tkbc_model_file}'.format(tkbc_model_file=args.tkbc_model_file))
else:
    tkbc_model = loadTkbcModel_complex('models/kg_embeddings/{tkbc_model_file}'.format(tkbc_model_file=args.tkbc_model_file))

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

train_split = 'train'
test = args.test
if args.model == 'bert' or args.model == 'roberta':
    qa_model = QA_lm(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name,args = args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split,dataset_name=args.dataset_name,args = args)
    test_dataset = QA_Dataset_Baseline(split=test,dataset_name=args.dataset_name,args = args)


elif args.model == 'embedkgqa':
    qa_model = QA_embedkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name,args = args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split,dataset_name=args.dataset_name,args = args)
    test_dataset = QA_Dataset_Baseline(split=test,dataset_name=args.dataset_name,args = args)


elif args.model == 'cronkgqa':
    qa_model = QA_cronkgqa(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name,args = args)
    valid_dataset = QA_Dataset_Baseline(split=args.eval_split,dataset_name=args.dataset_name,args = args)
    test_dataset = QA_Dataset_Baseline(split=test,dataset_name=args.dataset_name,args = args)


elif args.model == 'tempoqr':  # supervised models
    qa_model = QA_TempoQR(tkbc_model, args)
    dataset = QA_Dataset_TempoQR(split=train_split, dataset_name=args.dataset_name, args=args)
    valid_dataset = QA_Dataset_TempoQR(split=args.eval_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_TempoQR(split=test, dataset_name=args.dataset_name, args=args)


elif args.model == 'multiqa':
    dataset = QA_Dataset_MG(split=train_split, dataset_name=args.dataset_name, args = args)
    valid_dataset = QA_Dataset_MG(split=args.eval_split,dataset_name=args.dataset_name,args = args)
    test_dataset = QA_Dataset_MG(split=test,dataset_name=args.dataset_name, args = args)
    qa_model = QA_multiqa(tkbc_model,args)

else:
    print('Model %s not implemented!' % args.model)
    exit(0)

print('Model is', args.model)

if args.load_from != '':
    filename = 'models/qa_models/{model_file}.ckpt'.format(
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

result_filename = 'results/{model_file}.log'.format(
    model_file=args.save_to
)

train(qa_model, dataset, test_dataset, args, result_filename=result_filename)


print('Training finished')
