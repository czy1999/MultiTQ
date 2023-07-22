import pickle
import random
import torch
import numpy as np
from tcomplex import TComplEx
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def loadTkbcModel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model


def loadTkbcModel_complex(tkbc_model_file):
    print('Loading complex tkbc model from', tkbc_model_file)
    tcomplex_params = torch.load(tkbc_model_file)
    # complex_params = torch.load(tkbc_model_file)
    num_ent = tcomplex_params['embeddings.0.weight'].shape[0]
    num_rel = tcomplex_params['embeddings.1.weight'].shape[0]
    num_ts = tcomplex_params['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = tcomplex_params['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size

    # now put complex params in tcomplex model

    # tcomplex_params['embeddings.0.weight'] = complex_params['embeddings.0.weight']
    # tcomplex_params['embeddings.1.weight'] = complex_params['embeddings.1.weight']
    torch.nn.init.xavier_uniform_(tcomplex_params['embeddings.2.weight'])  # randomize time embeddings

    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(tcomplex_params)
    tkbc_model.cuda()
    print('Loaded complex tkbc model')
    return tkbc_model


def dataIdsToLiterals(d, all_dicts):
    new_datapoint = []
    id2rel = all_dicts['id2rel']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    wd_id_to_text = all_dicts['wd_id_to_text']
    new_datapoint.append(wd_id_to_text[id2ent[d[0]]])
    new_datapoint.append(wd_id_to_text[id2rel[d[1]]])
    new_datapoint.append(wd_id_to_text[id2ent[d[2]]])
    new_datapoint.append(id2ts[d[3]])
    new_datapoint.append(id2ts[d[4]])
    return new_datapoint


def getAllDicts(dataset_name, kg_dir):
    def readDict(filename,int_num = True):
        f = open(filename, 'r')
        d = {}
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 1:
                line.append('')  # in case literal was blank or whitespace
            if int_num:
                d[line[0]] = int(line[1])
            else:
                d[line[0]] = line[1]
        f.close()
        return d

    def getReverseDict(d):
        return {value: key for key, value in d.items()}

    if dataset_name == 'CronQuestions':
        base_path = '../data/{dataset_name}/{kg}/tkbc_processed_data/'.format(
            kg=kg_dir,
            dataset_name=dataset_name
        )
        dicts = {}
        for f in ['ent_id', 'rel_id', 'ts_id']:
            in_file = open(str(base_path + f), 'rb')
            dicts[f] = pickle.load(in_file)
        rel2id = dicts['rel_id']
        ent2id = dicts['ent_id']
        ts2id = dicts['ts_id']
        file_ent = '../data/{dataset_name}/{kg}/wd_id2entity_text.txt'.format(kg=kg_dir, dataset_name=dataset_name)
        file_rel = '../data/{dataset_name}/{kg}/wd_id2relation_text.txt'.format(kg=kg_dir, dataset_name=dataset_name)

        e = readDict(file_ent,int_num=False)
        r = readDict(file_rel,int_num=False)
        wd_id_to_text = dict(list(e.items()) + list(r.items()))

        id2rel = getReverseDict(rel2id)
        id2ent = getReverseDict(ent2id)
        id2ts = getReverseDict(ts2id)

        all_dicts = {'rel2id': rel2id,
                     'id2rel': id2rel,
                     'ent2id': ent2id,
                     'id2ent': id2ent,
                     'ts2id': ts2id,
                     'id2ts': id2ts,
                     'wd_id_to_text': wd_id_to_text
                     }
    else:
        base_path = '../data/{dataset_name}/{kg}/tkbc_processed_data'.format(kg=kg_dir, dataset_name=dataset_name)
        ent2id = readDict(base_path + '/ent_id')
        rel2id = readDict(base_path + '/rel_id')
        ts2id = readDict(base_path + '/ts_id')

        id2rel = getReverseDict(rel2id)
        id2ent = getReverseDict(ent2id)
        id2ts = getReverseDict(ts2id)
        all_dicts = {'rel2id': rel2id,
                     'id2rel': id2rel,
                     'ent2id': ent2id,
                     'id2ent': id2ent,
                     'ts2id': ts2id,
                     'id2ts': id2ts}
    return all_dicts


def checkQuestion(question, target_type):
    question_type = question['type']
    if target_type != question_type:
        return False
    return True


# def getDataPoint(question, all_dicts):

def predictTime(question, model, all_dicts, k=1):
    entities = list(question['entities'])
    times = question['time']
    target_type = 'simple_entity'
    ent2id = all_dicts['ent2id']
    rel2id = all_dicts['rel2id']
    ts2id = all_dicts['ts2id']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    head = ent2id[entities[0]]
    tail = ent2id[entities[1]]
    relation = question['relations']
    relation = rel2id[relation]  # + model.embeddings[1].weight.shape[0]//2 #+ 90
    data_point = [head, relation, tail, 1]
    data_batch = torch.from_numpy(np.array([data_point])).cuda()
    time_scores = model.forward_over_time(data_batch)
    val, ind = torch.topk(time_scores, k, dim=1)
    topk_set = set()
    for row in ind:
        for x in row:
            topk_set.add(id2ts[x.item()])
    return topk_set


def predictTail(question, model, all_dicts, k=1):
    entities = list(question['entities'])
    times = question['time']
    target_type = 'simple_entity'
    ent2id = all_dicts['ent2id']
    rel2id = all_dicts['rel2id']
    ts2id = all_dicts['ts2id']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    head = ent2id[entities[0]]
    time = ts2id[times[0]]
    relation = question['relations']
    relation = rel2id[relation]  # + model.embeddings[1].weight.shape[0]//2 #+ 90
    data_point = [head, relation, 1, time]
    data_batch = torch.from_numpy(np.array([data_point])).cuda()
    predictions, factors, time = model.forward(data_batch)
    val, ind = torch.topk(predictions, k, dim=1)
    topk_set = set()
    for row in ind:
        for x in row:
            topk_set.add(id2ent[x.item()])
    return topk_set


def checkIfTkbcEmbeddingsTrained(tkbc_model, split='test'):
    with open('../data/MultiTQ/questions/full_data/'+split+'.json') as f:
        questions = json.load(f)
    question_type ='equal'
    correct_count = 0
    total_count = 0
    k = 10  # hit at k
    all_dicts = getAllDicts('MultiTQ','kg')
    for i in tqdm(range(len(questions))):
        this_question_type = questions[i]['qtype']
        if question_type == this_question_type and questions[i]['answer_type'] == 'entity' and questions[i]['time_level'] == 'day':
            which_question_function = predictTail
        elif question_type == this_question_type and questions[i]['answer_type'] == 'time' and questions[i]['time_level'] == 'day':
            which_question_function = predictTime
        else:
            continue
        total_count += 1
        id = i
        predicted = which_question_function(questions[id], tkbc_model, all_dicts, k)
        intersection_set = set(questions[id]['answers']).intersection(predicted)
        if len(intersection_set) > 0:
            correct_count += 1
    print(question_type, correct_count, total_count, correct_count / total_count)



def save_model(qa_model, filename):
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return


def print_info(args):
    print('#######################')
    print('Model: ' + args.model)
    print('TKG Embeddings: ' + args.tkbc_model_file)
    print('TKG for QA (if applicable): ' + args.tkg_file)
    print('#######################')


def eval_cron(qa_model, dataset, batch_size=128, split='valid', k=10):
    num_workers = 4
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k  # not change name in fn signature since named param used in places
    k_list = [1]
    # k_list = [1,2,5, 10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1]  # last one assumed to be target
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['type']
            if 'simple' in question_type:
                simple_complex_type = 'simple'
            else:
                simple_complex_type = 'complex'
            entity_time_type = question['answer_type']
            # question_type = question['template']
            predicted = topk_answers[i][:k]
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0
                # eval_log.append(question)
                # eval_log.append(predicted[0])
            question_types_count[question_type].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1

        eval_accuracy = hits_at_k / total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, entity_time_count]:
            # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value) / len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type=key,
                    hits_at_k=round(hits_at_k, 3),
                    num_questions=len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log


def eval_multi(qa_model, dataset, batch_size=128, split='valid', k=10):
    num_workers = 4
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k  # not change name in fn signature since named param used in places
    k_list = [1, 10]
    # k_list = [1,2,5, 10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1]  # last one assumed to be target
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    # do eval for each k in k_list
    # want multiple hit@k
    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)
        time_level_count = defaultdict(list)
        time_level_count_all = defaultdict(list)
        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['qtype']
            time_level = question['time_level']
            if 'Single' in question['qlabel']:
                simple_complex_type = 'Single'
            else:
                simple_complex_type = 'Multiple'
            entity_time_type = question['answer_type']
            # question_type = question['template']
            predicted = topk_answers[i][:k]

            # multiple time answers - hard way
            if question['answer_type']=='time':
                if len(question['answers'][0]) == 4:
                    predicted = [x[:4] for x in predicted]
                elif len(question['answers'][0]) == 7:
                    predicted = [x[:7] for x in predicted]

            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                val_to_append = 0

            question_types_count[question_type].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count_all[time_level].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count[question['qtype'] + '-' + time_level].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1

        eval_accuracy = hits_at_k / total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

        time_level_count_all = dict(sorted(time_level_count_all.items(), key=lambda x: x[0].lower()))
        time_level_count = dict(sorted(time_level_count.items(), key=lambda x: x[0].lower()))
        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, time_level_count_all, time_level_count,
                           entity_time_count]:
            # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value) / len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type=key,
                    hits_at_k=round(hits_at_k, 3),
                    num_questions=len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log


def append_log_to_file(eval_log, epoch, filename):
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()


def train(qa_model, dataset, valid_dataset, args, result_filename=None):
    if args.dataset_name == 'CronQuestions':
        eval = eval_cron
    else:
        eval = eval_multi
    num_workers = 5
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=dataset._collate_fn)
    max_eval_score = 0
    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = 'results/{dataset_name}/{model_file}.log'.format(
            dataset_name=args.dataset_name,
            model_file=args.save_to
        )
    checkpoint_file_name = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        model_file=args.save_to
    )

    # if not loading from any previous file
    # we want to make new log file
    # also log the config ie. args to the file
    if args.load_from == '':
        print('Creating new log file')
        f = open(result_filename, 'a+')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Config: \n')
        for key, value in vars(args).items():
            key = str(key)
            value = str(value)
            f.write('%s:\t%s\n' % (key, value))
        f.write('\n')
        f.close()

    max_eval_score = 0.

    print('Starting training')
    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for i_batch, a in enumerate(loader):
            qa_model.zero_grad()
            # so that don't need 'if condition' here
            # scores = qa_model.forward(question_tokenized.cuda(),
            #             question_attention_mask.cuda(), entities_times_padded.cuda(),
            #             entities_times_padded_mask.cuda(), question_text)

            answers_khot = a[-1]  # last one assumed to be target
            scores = qa_model.forward(a)
            loss = qa_model.loss(scores, answers_khot.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, args.max_epochs))
            loader.update()

        print('Epoch loss = ', epoch_loss)
        if (epoch + 1) % args.valid_freq == 0:
            print('Starting eval')
            eval_score, eval_log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size,
                                        split=args.eval_split, k=args.eval_k)
            if eval_score > max_eval_score:
                print('Valid score increased')
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            # log each time, not max
            # can interpret max score from logs later
            append_log_to_file(eval_log, epoch, result_filename)
