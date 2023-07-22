import difflib
import json
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from tqdm.contrib import tenumerate
import re
from dateutil.parser import parse


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
    with open(kg_path,'r',encoding='utf-8') as f:
        data = f.readlines()
    all_entities = []
    for i in data:
        all_entities.append(i.strip().split('\t')[0])
    keys = all_entities
    tagger = SequenceTagger.load("flair/ner-english-large")


    for d in ['test','dev','train']:
        print(d.upper() + ' DATASET PROCESSING......')
        with open(input_path+d+'.json','r',encoding='utf-8') as f:
            dataset = json.load(f)
        questions = [Sentence(x['question']) for x in dataset]
        for ix,s in tqdm(tenumerate(questions)):
            tagger.predict(s)
            e = []
            for entity in s.get_spans('ner'):
                entity_text = difflib.get_close_matches(entity.text,keys,n=1)
                e.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
                clean_ner_result = [y for y in e if len(y['entity'])>0]
            dataset[ix]['time'] = extract_time(dataset[ix]['question'])
            dataset[ix]['entities'] = [x['entity'][0] for x in clean_ner_result]
            dataset[ix]['entity_positions'] = clean_ner_result
        with open(output_path + d + '.json', 'w',encoding='utf-8') as obj:
            obj.write(json.dumps(dataset, indent=4,ensure_ascii=False))
        print(d.upper() + ' DATASET PROCESSED')

