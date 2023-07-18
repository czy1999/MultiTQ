import argparse
import logging
import pickle
import math
import random
import torch
from torch import optim
from qa_baselines import QA_cronkgqa_RT,QA_timeQA2,QA_cronkgqa_soft,QA_baseline, QA_lm, QA_embedkgqa, QA_cronkgqa, QA_MG_transformer,QA_MG_selected_transformer,QA_MG_mean,QA_multiqa
from qa_tempoqr import QA_TempoQR
from qa_datasets import QA_Dataset, QA_Dataset_Baseline, QA_Dataset_MG,QA_Dataset_timeQA
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info, save_model, append_log_to_file,eval,train
from collections import defaultdict
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from torch import nn
import pandas as pd

import re
import numpy as np
from tcomplex import TComplEx
from transformers import RobertaModel
from transformers import BertModel
from transformers import DistilBertModel
import pdb
from torch.nn import LayerNorm
from utils import loadTkbcModel, loadTkbcModel_complex, print_info
import utils
import difflib
from flair.data import Sentence
from flair.models import SequenceTagger

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time 
import streamlit as st
from pylab import *
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False 

def extract_time(question_text):
    endings=['st','nd','rd']+17*['th']+['st','nd','rd']+7*['th']+['st']
    days = [str(x+1)+endings[x] for x in range(31)]
    months=['January','February','March','April', 'May','June','July','August','September','October','Novmber','December']
    months_abbr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep', 'Oct', 'Nov','Dec']

    day = -1
    month = -1
    year = -1

    # day
    for i in range(32):
        if ' '+ str(i)+',' in question_text:
            day =i

    for i in range(31):
        if ' ' + days[i] in question_text:
            day =i+1

    # month 
    for i in range(12):
        if ' ' + months[i] in question_text or ' ' + months_abbr[i] in question_text:
            month =i+1

    # year
    reg = '\d\d\d\d'
    r= re.search(reg,question_text)
    if r!=None:
        year = question_text[r.start():r.end()]

    reg = '\d\d\d\d-\d\d'
    r= re.search(reg,question_text)
    if r!=None:
        t= question_text[r.start():r.end()]
        year = t.split('-')[0]
        month = t.split('-')[1]

    reg = '\d\d\d\d-\d\d-\d\d'
    r= re.search(reg,question_text)
    if r!=None:
        t= question_text[r.start():r.end()]
        year = t.split('-')[0]
        month = t.split('-')[1]
        day = t.split('-')[2]
        
    else:
        reg = '\d\d\d\d-\d\d-\d'
        r= re.search(reg,question_text)
        if r!=None:
            t= question_text[r.start():r.end()]
            year = t.split('-')[0]
            month = t.split('-')[1]
            day = t.split('-')[2]

    if year == -1:
        return []
    elif day!=-1:
        month = int(month)
        day = int(day)
        month_text = '0'+str(month) if month<10 else str(month)
        day_text = '0'+str(day) if day<10 else str(day)
        return [str(year)+'-'+month_text+'-'+day_text]
    elif month!=-1:
        month = int(month)
        month_text = '0'+str(month) if month<10 else str(month)
        return [str(year)+'-'+month_text]
    else:
        return [str(year)]

def extract_entity(question_text):
    NER_result = []
    NEL_result = []
    s = Sentence(question_text)
    tagger.predict(s)
    for entity in s.get_spans('ner'):
        entity_text = difflib.get_close_matches(entity.text,keys,n=1)
        NER_result.append(entity)
        NEL_result.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
    return NEL_result  

def prepare_question(question_text):
    q = {'question':question_text}
    NEL_result =  extract_entity(question_text)
    q['entities'] = [x['entity'][0] for x in NEL_result]
    q['entity_positions'] = NEL_result
    q['time'] = extract_time(question_text)

    # Just for preprocess
    q['answers'] = ['Aceh']
    q['answer_type'] = 'entity'
    prepare_q = valid_dataset.prepare_data_([q])
    prepare_q = [prepare_q['question_text'][0], prepare_q['head'][0], prepare_q['tail'][0], prepare_q['time'][0], prepare_q['answers_arr'][0]]
    prepare_q = valid_dataset._collate_fn([prepare_q])
    return prepare_q,q['entity_positions'],q['time']

@st.cache(allow_output_mutation=True)
def load_models():
    tkbc_model = loadTkbcModel('models/kg_embeddings/{tkbc_model_file}'.format(
        tkbc_model_file=args.tkbc_model_file
    ))
    tagger = SequenceTagger.load("flair/ner-english-large")
    valid_dataset = QA_Dataset_MG(split='dev')
    keys = list(valid_dataset.all_dicts['ent2id'].keys())
    return tkbc_model,keys,tagger,valid_dataset

class A():
    def __init__(self):
        self.model = 'bert'
        self.lm_frozen = 1
        self.frozen = 1
        self.tkbc_model_file = 'tkbc_tcomplex_256.ckpt'
        self.eval_split = 'test'
        self.lr = 1e-3
        self.valid_batch_size = 10
        self.eval_split = 'valid'
        self.batch_size = 128
        self.load_from = ''
        self.max_epochs = 50
        self.save_to = 'qa_lm'
        self.valid_freq = 1
        self.eval_k = 10
args = A()

import json


st.title("DEMO")
st.image('https://s1.ax1x.com/2022/10/21/x6bkKU.png')

col1, col2, col3,col4 = st.columns(4)
col1.metric(label="Hits@1", value=0.366, delta=+0.022,
    delta_color="inverse")
col2.metric(label="Hits@10", value=0.659, delta=+0.010,
    delta_color="inverse")
col3.metric(label="month(Hits@1)", value=0.58, delta=+0.041,
    delta_color="inverse")
col4.metric(label="year(Hits@1)", value=0.65, delta=+0.052,
    delta_color="inverse")
with open('../data/questions/test.json') as f:
    j = json.load(f)
st.json(j[:10])
tkbc_model,keys,tagger,valid_dataset = load_models()
qa_model = QA_multiqa(tkbc_model,args)
qa_model.load_state_dict(torch.load('./models/qa_models/QA_multiqa_ner_v1_1e-3.ckpt'))
qa_model = qa_model.cuda()
qa_model.eval()

st.subheader('Try with any question')
question_text = st.text_input('question:', 'Who visited Japan before China?')
st.write('The current movie question is：', question_text)

prepare_q,entity,time = prepare_question(question_text)
score = qa_model.forward(prepare_q)
pred = valid_dataset.getAnswersFromScores(score[0], k=10)
sorted_score = score.sort(descending=True)[0][0].detach().cpu().numpy()[:10]

st.write('question:',question_text)
st.text('NEL:')
for i in entity:
    st.write(i,end = ' ')
 
if len(time)>0:
    st.write('Time:',time[0])
else:
    st.write('Time: Not Found!')
result = pd.DataFrame({'Answer':pred,'Score':sorted_score})
st.write(result)


