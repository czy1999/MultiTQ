from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List
import json

import numpy as np
import torch
# from qa_models import QA_model
import utils
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import DistilBertTokenizer,AlbertTokenizer
from transformers import BertTokenizer
import random
from torch.utils.data import Dataset, DataLoader
# from nltk import word_tokenize
# warning: padding id 0 is being used, can have issue like in Tucker
# however since so many entities (and timestamps?), it may not pose problem
import pdb
from copy import deepcopy
from collections import defaultdict
import random

class QA_Dataset_cron(Dataset):
    def __init__(self,
                split,
                dataset_name,
                args,
                tokenization_needed=True):
        filename = '../data/{dataset_name}/questions/{split}.pickle'.format(
            dataset_name=dataset_name,
            split=split
        )
        questions = pickle.load(open(filename, 'rb'))

        #probably change for bert/roberta?
        self.tokenizer_class = DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',mirror = 'tuna')
        if args.lm_model == 'albert':
            self.pretrained_weights = 'albert-base-v2'
            self.tokenizer = AlbertTokenizer.from_pretrained(self.pretrained_weights)
        # filepath = './saved_pkl/tokenizer.pkl'
        # with open(filepath, 'wb') as f:
        #     pickle.dump(self.tokenizer, f)
        # filepath = './tokenizer.pkl'
        # with open(filepath, 'rb') as f:
        #     self.tokenizer = pickle.load(f)
        self.all_dicts = utils.getAllDicts(dataset_name, kg_dir=args.kg_dir)
        print('Total questions = ', len(questions))
        self.data = questions
        self.tokenization_needed = tokenization_needed


    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entities']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        question_text = question['question']
        times = question['times']
        ts2id = self.all_dicts['ts2id']
        loc_time = []
        for t in times:
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id']) # add num entities
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        # todo: cant do len == 4 since 3 digit times also there
        if 'Q' in s:
            return False
        elif s.isdigit():
            return True
        else:
            return False

    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id


    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output

    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'

    def getEntityToText(self, entity_wd_id):
        return self.all_dicts['wd_id_to_text'][entity_wd_id]

    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return self.getEntityToText(ent)

    def getEntityIdToWdId(self, id):
        return self.all_dicts['id2ent'][id]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            output.append(ts2id[(t, 0, 0)])
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return answers

    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return s, answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored 
    # while the position with the value of False will be unchanged.
    # 
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len = -1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool) # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False # fills good area with False
        return out_tensor, mask

    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text,
                'entity_time_ids': entity_time_ids,
                'answers_arr': answers_arr}

    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1]) # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = [] # we want for each paraphrase
            template = question['template']
            #for nl_question in question['paraphrases']:
            nl_question =  question['paraphrases'][0]
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
            #print(keyword_dicts)
            #print(template, nl_question)
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output

class QA_Dataset_Baseline_cron(QA_Dataset_cron):
    def __init__(self, split, dataset_name, args, tokenization_needed=True):
        super().__init__(split, dataset_name, args,tokenization_needed)
        print('Preparing data for split %s' % split)
        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()



        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times


    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered=[]
        # self.data=[]
        for i,question in enumerate(data):
            self.data_ids_filtered.append(i)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]


            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in entities_list_with_locations] # ordering necessary otherwise set->list conversion causes randomness
            head = entities[0] # take an entity
            if len(entities) > 1:
                tail = entities[1]
            else:
                tail = entities[0]
            times_in_question = question['times']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0] # take a time. if no time then 0
                # exit(0)
            else:
                time = 0


            time += num_total_entities
            heads.append(head)
            times.append(time)
            tails.append(tail)
            question_text.append(q_text)

            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)

        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data=[self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}
    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single #,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single
    def get_dataset_ques_info(self):
        type2num={}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]]=0
            type2num[question["type"]]+=1
        return {"type2num":type2num, "total_num":len(self.data_ids_filtered)}.__str__()

class QA_Dataset_multi(Dataset):
    def __init__(self,
                 split,
                 dataset_name='MultiTQ',
                 tokenization_needed=True, args=None):
        filename = '../data/{dataset_name}/questions/{sub_dataset}/{split}.json'.format(dataset_name=dataset_name,sub_dataset=args.sub_dataset, split=split)

        with open(filename, 'r') as obj:
            questions = json.load(obj)
        # questions = [x for x in questions if x['qtype'] == 'equal' and x['time_level'] == 'day']
        # questions = [x for x in questions if x['qlabel'] == 'Multiple' and x['time_level'] == 'day']


        self.tokenizer_class = DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',mirror = 'tuna')
        if args.lm_model == 'albert':
            self.pretrained_weights = 'albert-base-v2'
            self.tokenizer = AlbertTokenizer.from_pretrained(self.pretrained_weights)

        self.all_dicts = utils.getAllDicts(dataset_name,args.kg_dir)
        print('Total questions = ', len(questions))
        self.data = questions
        self.tokenization_needed = tokenization_needed

    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entity_positions']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e['entity'][0]]
            location = e['position'][0]
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        loc_time = []
        question_text = question['question']
        if len(question['time']) != 0:
            time = question['time'][0]
            ts2id = self.all_dicts['ts2id']
            if time in ts2id.keys():
                t_ids = [ts2id[time] + len(self.all_dicts['ent2id'])]
            else:
                keys = [x for x in ts2id.keys() if x.startswith(time)]
                t_ids = [ts2id[key] + len(self.all_dicts['ent2id']) for key in keys]
            location = question_text.find(time)
            for t_id in t_ids:
                loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        # todo: cant do len == 4 since 3 digit times also there
        if '20' in s or s.isdigit():
            return True
        else:
            return False


    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[t] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id

    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e.replace('_',' ')])
        return output

    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'

    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return ent

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        # hard write
        if times[0] in ts2id.keys():
            return [ts2id[times[0]]]
        for t in times:
            keys = [x for x in ts2id.keys() if x.startswith(t)]
            output = [ts2id[key] for key in keys]
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToText(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time)
        return answers

    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToText(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time)
        return s, answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored
    # while the position with the value of False will be unchanged.
    #
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len=-1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool)  # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False  # fills good area with False
        return out_tensor, mask

    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['question']
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text,
                'entity_time_ids': entity_time_ids,
                'answers_arr': answers_arr}

    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1])  # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = []  # we want for each paraphrase
            template = question['template']
            # for nl_question in question['paraphrases']:
            nl_question = question['question']
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
            # print(keyword_dicts)
            # print(template, nl_question)
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output

class QA_Dataset_Baseline_muti(QA_Dataset_multi):
    def __init__(self, split, dataset_name='MultiTQ', tokenization_needed=True,args = None):
        super().__init__(split, dataset_name, tokenization_needed,args)
        print('Preparing data for split %s' % split)

        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = self.all_dicts['id2ts']
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()

        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times

    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered = []
        # self.data=[]
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['question']

            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            if len(entities) == 0:
                head = 0
                tail = 0
            else:
                head = entities[0]  # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                else:
                    tail = entities[0]
            times_in_question = question['time']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)  # take a time. if no time then 0
                # exit(0)
            else:
                time = [0]

            time = [int(x) + num_total_entities for x in time]
            heads.append(int(head))
            times.append(time)
            tails.append(int(tail))
            question_text.append(q_text)

            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [int(x) + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)

        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}

    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single  # ,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        # times = [item[3] for item in items]

        # times = torch.from_numpy(np.array([item[3][0] for item in items]))
        times = torch.from_numpy(np.array([np.random.choice(item[3]) for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single

    def get_dataset_ques_info(self):
        type2num = {}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]] = 0
            type2num[question["type"]] += 1
        return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()

class QA_Dataset_MultiQA_muti(QA_Dataset_multi):
    def __init__(self, split, dataset_name='MultiTQ', tokenization_needed=True,args = None):
        super().__init__(split, dataset_name, tokenization_needed, args)
        print('Preparing data for split %s' % split)
        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = self.all_dicts['id2ts']
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()

        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times

    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered = []
        # self.data=[]
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)
            q_text = question['question']
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            if len(entities) == 0:
                head = 0
                tail = 0
            else:
                head = entities[0]  # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                else:
                    tail = entities[0]
            times_in_question = question['time']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)  # take a time. if no time then 0
                # exit(0)
            else:
                time = [0]

            time = [x + num_total_entities for x in time]
            heads.append(head)
            times.append(time)
            tails.append(tail)
            question_text.append(q_text)

            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)

        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}

    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single  # ,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = [torch.tensor(item[3]) for item in items]
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single

    def get_dataset_ques_info(self):
        type2num = {}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]] = 0
            type2num[question["type"]] += 1
        return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()