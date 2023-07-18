from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-large")


%%time
questions = [Sentence(x['question']) for x in test_dataset.data]
tagger.predict(questions)


%%time
keys = list(test_dataset.all_dicts['ent2id'].keys())
ner_results = []
for s in questions:
    e = []
    for entity in s.get_spans('ner'):
        entity_text = difflib.get_close_matches(entity.text,keys,n=1)
        e.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
    ner_results.append(e)
with open('test_ner_results.pkl','wb') as f:
    pickle.dump(ner_results,f)
    
    
    
%%time
questions = [Sentence(x['question']) for x in valid_dataset.data]
tagger.predict(questions)
keys = list(valid_dataset.all_dicts['ent2id'].keys())
ner_results = []
for s in questions:
    e = []
    for entity in s.get_spans('ner'):
        entity_text = difflib.get_close_matches(entity.text,keys,n=1)
        e.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
    ner_results.append(e)
with open('valid_ner_results.pkl','wb') as f:
    pickle.dump(ner_results,f)
    
    
    
%%time
questions = [x['question'] for x in dataset.data]
keys = list(valid_dataset.all_dicts['ent2id'].keys())
ner_results = []
for q in questions:
    s = Sentence(q)
    tagger.predict(s)
    e = []
    for entity in s.get_spans('ner'):
        entity_text = difflib.get_close_matches(entity.text,keys,n=1)
        e.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
    ner_results.append(e)
with open('train_ner_results.pkl','wb') as f:
    pickle.dump(ner_results,f)
    
    
s = 'train'
test = dataset.data

with open(s+'_ner_results.pkl','rb') as f:
    test_ner_results = pickle.load(f)
clean_test_ner_results = [[y for y in x if len(y['entity'])>0] for x in test_ner_results]
for i,q in enumerate(test):
    q['entities'] = [x['entity'][0] for x in clean_test_ner_results[i]]
    q['entity_positions'] = clean_test_ner_results[i]
for x in test:
    try:
        x.pop('template')
        x.pop('relations')
        x.pop('type')
    except:
        pass
with open('../data/questions/processed_questions/train.json', 'w') as obj:
    obj.write(json.dumps(test, indent=4))