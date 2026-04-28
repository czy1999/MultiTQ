import difflib
import json
import os
import multiprocessing as mp
from functools import lru_cache
import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from tqdm.contrib import tenumerate
import re
from dateutil.parser import parse

flair.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


_KEYS = None
_KEYS_SET = None


def _worker_init(keys_list, keys_set):
    """Pool initializer：每个 worker 进程加载一次 KG 实体表。"""
    global _KEYS, _KEYS_SET
    _KEYS = keys_list
    _KEYS_SET = keys_set


def _fuzzy_match(text):
    """单条实体文本到 KG 实体名的最近匹配。"""
    if text in _KEYS_SET:
        return text, (text,)
    m = difflib.get_close_matches(text, _KEYS, n=1)
    return text, tuple(m)


def extract_time(sentence: str) -> str:
    date_pattern = r'\b(?:\d{1,2}\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s?\d{1,2}?(?:\s|,)?\s?\d{2,4}\b|\b\d{4}\b'
    date_match = re.search(date_pattern, sentence)

    if date_match:
        date_str = date_match.group()
        date_obj = parse(date_str)
        date_format = ''

        if len(date_str.split(' ')) == 3:
            date_format = f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"
        elif len(date_str.split(' ')) == 2:
            date_format = f"{date_obj.year}-{date_obj.month:02d}"
        elif len(date_str.split(' ')) == 1:
            date_format = f"{date_obj.year}"

        return [date_format]
    else:
        return []
    


if __name__ == '__main__':
    kg_path = '../data/MultiTQ/kg/tkbc_processed_data/ent_id'
    input_path = '../data/MultiTQ/questions/'
    output_path = '../data/MultiTQ/questions/processed_questions/'
    os.makedirs(output_path, exist_ok=True)

    with open(kg_path,'r',encoding='utf-8') as f:
        data = f.readlines()
    all_entities = []
    for i in data:
        all_entities.append(i.strip().split('\t')[0])
    keys = all_entities
    keys_set = set(keys)

    print('Loading flair NER model on', flair.device)
    tagger = SequenceTagger.load("flair/ner-english-large")

    BATCH_SIZE = 64
    NUM_WORKERS = max(1, (os.cpu_count() or 4) - 2)
    print(f'Will use {NUM_WORKERS} CPU workers for fuzzy matching')

    for d in ['test', 'dev', 'train']:
        print(f'\n========== {d.upper()} DATASET ==========')
        with open(input_path + d + '.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f'  total questions: {len(dataset)}')

        # ---- 1) Flair NER 批预测（GPU） ----
        questions = [Sentence(x['question']) for x in dataset]
        for start in tqdm(range(0, len(questions), BATCH_SIZE),
                          desc=f'  [{d}] NER', unit='batch'):
            tagger.predict(questions[start:start + BATCH_SIZE],
                           mini_batch_size=BATCH_SIZE)

        # ---- 2) 收集所有 NER 输出 + 去重实体文本 ----
        per_q_spans = []  # [[(text, start, end), ...], ...]
        unique_texts = set()
        for s in questions:
            spans = []
            for ent in s.get_spans('ner'):
                spans.append((ent.text, ent.start_position, ent.end_position))
                unique_texts.add(ent.text)
            per_q_spans.append(spans)
        unique_texts = sorted(unique_texts)
        print(f'  unique entity texts to match: {len(unique_texts)}')

        # ---- 3) 多进程 difflib 模糊匹配 ----
        text_to_match = {}
        if unique_texts:
            ctx = mp.get_context('fork')  # Linux 下 fork 共享只读内存
            with ctx.Pool(processes=NUM_WORKERS,
                          initializer=_worker_init,
                          initargs=(keys, keys_set)) as pool:
                chunksize = max(1, len(unique_texts) // (NUM_WORKERS * 8))
                for text, matched in tqdm(
                        pool.imap_unordered(_fuzzy_match, unique_texts,
                                            chunksize=chunksize),
                        total=len(unique_texts),
                        desc=f'  [{d}] match', unit='ent'):
                    text_to_match[text] = matched

        # ---- 4) 把匹配结果写回 dataset ----
        for ix, spans in enumerate(tqdm(per_q_spans, desc=f'  [{d}] write',
                                        unit='q')):
            e = []
            for text, start, end in spans:
                matched = list(text_to_match.get(text, ()))
                e.append({'entity': matched, 'position': [start, end]})
            clean_ner_result = [y for y in e if len(y['entity']) > 0]
            dataset[ix]['time'] = extract_time(dataset[ix]['question'])
            dataset[ix]['entities'] = [x['entity'][0] for x in clean_ner_result]
            dataset[ix]['entity_positions'] = clean_ner_result

        with open(output_path + d + '.json', 'w', encoding='utf-8') as obj:
            obj.write(json.dumps(dataset, indent=4, ensure_ascii=False))
        print(f'  [{d}] DONE -> {output_path + d + ".json"}')

