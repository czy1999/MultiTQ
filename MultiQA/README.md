# Multitq
python train_qa_model.py --tkbc_model_file icews.ckpt --model bert --save_to bert --kg_dir kg --dataset_name MultiTQ --sub_dataset processed_questions

python train_qa_model.py --tkbc_model_file icews.ckpt --model cronkgqa --save_to cronkgqa --kg_dir kg --dataset_name MultiTQ --sub_dataset processed_questions

python train_qa_model.py --tkbc_model_file icews.ckpt --model multiqa_mean --save_to multiqa --kg_dir kg --dataset_name MultiTQ --sub_dataset processed_questions


python train_qa_model.py --tkbc_model_file enhanced_kg_v998.ckpt --model bert --save_to bert_enhanced --kg_dir enahnced_kg --dataset_name MultiTQ --sub_dataset processed_questions

python train_qa_model.py --tkbc_model_file enhanced_kg_v998.ckpt --model cronkgqa --save_to cronkgqa_enhanced --kg_dir enahnced_kg --dataset_name MultiTQ --sub_dataset processed_questions

python train_qa_model.py --tkbc_model_file enhanced_kg_v998.ckpt --model multiqa_mean --save_to multiqa_enhanced --kg_dir enahnced_kg --dataset_name MultiTQ --sub_dataset processed_questions


with time 

python train_qa_model.py --tkbc_model_file enhanced_kg_with_time.ckpt --model cronkgqa --save_to cronkgqa_enhanced_with_time --kg_dir enhanced_kg_with_time --dataset_name MultiTQ --sub_dataset processed_questions


mr
python train_qa_model.py --tkbc_model_file icews.ckpt --model mr --save_to mr --kg_dir kg --dataset_name MultiTQ --sub_dataset processed_questions


# CronKGQA

python train_qa_model.py --tkbc_model_file tcomplex.ckpt --model bert --save_to bert --kg_dir kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner

python train_qa_model.py --tkbc_model_file tcomplex.ckpt --model cronkgqa --save_to cronkgqa --kg_dir kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner

python train_qa_model.py --tkbc_model_file tcomplex.ckpt --model multiqa_mean --save_to multiqa --kg_dir kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner


python train_qa_model.py --tkbc_model_file enhanced_wiki_v9587.ckpt --model bert --save_to bert_enhanced --kg_dir enhanced_kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner

python train_qa_model.py --tkbc_model_file enhanced_wiki_v9587.ckpt --model cronkgqa --save_to cronkgqa_enhanced --kg_dir enhanced_kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner

python train_qa_model.py --tkbc_model_file enhanced_wiki_v9587.ckpt --model multiqa_mean --save_to multiqa_enhanced --kg_dir enhanced_kg --dataset_name CronQuestions --sub_dataset processed_questions_new_ner
`

