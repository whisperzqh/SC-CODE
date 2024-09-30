
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
data_path = '~/scientific_pl/code_to_text_data/code_nl_data/'
doc_data_path = '~/scientific_pl/code_to_text_data/code_doc/'
import random
from nltk.translate.bleu_score import sentence_bleu
from bleu import compute_bleu
import numpy as np
truncate_pattern_julia = [r"\n\n^#", "^'''", "\n\n\n"]
truncate_pattern_r = [r"\n\n^#","\n\n\n"]
truncate_pattern_matlab = [r"\n\n^%", "^%{", "\n\n\n"]

#top k
'''
generate prompts by retrieval from the training corpus
'''
from rank_bm25 import BM25Okapi

def get_similar_prompts(lang, data, prompt_num, bm25, query, data_type='docstring'):
    # data_type: code or docstring
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    # get [prompt_num] most similar prompts
    prompt_idx = np.argpartition(doc_scores, -prompt_num)[-prompt_num:]
    prompt = ""
    for idx in prompt_idx:
        # code = json.loads(data[idx].strip())['code'] 
        # docstring = json.loads(data[idx].strip())['docstring']
        code = json.loads(data[idx].strip())['code_string'] 
        docstring = json.loads(data[idx].strip())['des_string']
        if lang == 'matlab':
            prompt += '% ' + docstring + '\n' + code + '\n\n'
        else:
            prompt += '# ' + docstring + '\n' + code + '\n\n'
    if lang == 'matlab':
        prompt += "% <insert>\n"
    else:
        prompt += "# <insert>\n"
    return prompt


def generate_batch_data(lang, split, bz):
    data = open(data_path + f'{lang}/{split}.jsonl').readlines()
    all_code = []
    all_docstring = []
    for line in data:
        code = json.loads(line)['code']
        docstring = json.loads(line)['docstring']
        all_code.append(code)
        all_docstring.append(docstring)
    batch_len = len(all_code) // bz
    for i in range(batch_len):
        yield all_code[i*bz:(i+1)*bz], all_docstring[i*bz:(i+1)*bz]

def add_prompt_text2code(docstring_batch, lang, prompt_num, prompt_type='template', data_type='docstring'):
    prompt = []
    if prompt_type == 'retrieval': 
        data = open(data_path + f'{lang}/train.jsonl').readlines()
        corpus = [json.loads(item)[data_type] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    elif prompt_type == 'retrieval_doc': 
        data = open(doc_data_path + f'{lang}_docu_des_final.json').readlines()
        if data_type == 'docstring':
            corpus = [json.loads(item)['des_string'] for item in data]
        else:
            copus = [json.loads(item)[data_type] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    for docstring in docstring_batch:
        if prompt_type == 'zero-shot':
            if lang == 'matlab':
                example = "% <insert>\n"
            else:
                example = "# <insert>\n"
            prompt.append(example.replace('<insert>', docstring.strip()))
        elif prompt_type == 'template':
            example = read_examples(lang, prompt_num)
            prompt.append(example.replace('<insert>', docstring.strip()))
        else:
            retrieved_prompt = get_similar_prompts(lang, data, prompt_num, bm25, docstring, data_type)
            prompt.append(retrieved_prompt.replace('<insert>', docstring.strip()))

    return prompt

def read_examples(lang, num):
    data = open(f'../template_prompt/prompt_{lang}.jsonl','r').readlines()
    prompt = ""
    for line in data[:num]:
        code = json.loads(line)['code'] 
        docstring = json.loads(line)['docstring']
        if lang == 'matlab':
            prompt += '% ' + docstring + '\n' + code + '\n\n'
        else:
            prompt += '# ' + docstring + '\n' + code + '\n\n'
    if lang == 'matlab':
        prompt += "% <insert>\n"
    else:
        prompt += "# <insert>\n"
    return prompt




def batch_generate_text2code(lang, batch_size, prompt_num, prompt_type, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi").cuda()
    tokenizer.pad_token = tokenizer.eos_token
    bleu_score = []
    for code, docstring in generate_batch_data(lang, 'test', batch_size):
        prompt = add_prompt_text2code(docstring, lang, prompt_num, prompt_type=prompt_type)
        inputs = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt").to(0)
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]
        sample = model.generate(input_ids=input_ids, attention_mask=attn_masks, max_length=max_length)
        for i in range(len(sample)):
            generated_code = tokenizer.decode(sample[i][len(input_ids[i]):], truncate_before_pattern=eval(f'truncate_pattern_{lang}'), skip_special_tokens=True)
            try:
                bleu = compute_bleu([code[i].strip().split()], generated_code.strip().split())[0]
                bleu_score.append(bleu)
                print(f'bleu score: {bleu}')
            except:
                pass
    return bleu_score


def get_bleu_score(lang, batch_size, prompt_num, max_length, log_file, prompt_type='template'):
    with open(log_file, 'a') as f:
        f.write(f'{prompt_type} prompt, lang: {lang}, prompt_num: {prompt_num}, max_length: {max_length} ')
        f.flush()
        bleu_score = batch_generate_text2code(lang, batch_size, prompt_num, prompt_type, max_length)
        print(f'average bleu score for {lang}: {sum(bleu_score)/len(bleu_score)}')
        f.write(f'average bleu score for {lang}: {sum(bleu_score)/len(bleu_score)} \n')
        f.flush()


# retrieval_doc
# get_bleu_score('r', 4, 2, 1024, 'scientific_codegen_log.txt', 'retrieval_doc')
# get_bleu_score('matlab', 4, 2, 1024, 'scientific_codegen_log.txt','retrieval_doc')
# get_bleu_score('julia', 4, 2, 1024, 'scientific_codegen_log.txt','retrieval_doc')
get_bleu_score('r', 4, 1, 1024, 'scientific_codegen_log.txt', 'retrieval_doc')
get_bleu_score('matlab', 4, 1, 1024, 'scientific_codegen_log.txt','retrieval_doc')
get_bleu_score('julia', 4, 1, 1024, 'scientific_codegen_log.txt','retrieval_doc')


# get_bleu_score('r', 4, 2, 1024, 'scientific_codegen_log.txt', 'retrieval')
# get_bleu_score('matlab', 4, 2, 1024, 'scientific_codegen_log.txt','retrieval')
# get_bleu_score('julia', 4, 2, 1024, 'scientific_codegen_log.txt','retrieval')


# get_bleu_score('r', 4, 2, 1024, 'scientific_codegen_log.txt', 'template')
# get_bleu_score('matlab', 4, 2, 1024, 'scientific_codegen_log.txt','template')
# get_bleu_score('julia', 4, 2, 1024, 'scientific_codegen_log.txt','template')

# get_bleu_score('r', 4, 0, 1024, 'scientific_codegen_log.txt','zero-shot')
# get_bleu_score('matlab', 4, 0, 1024, 'scientific_codegen_log.txt','zero-shot')
# get_bleu_score('julia', 4, 0, 1024, 'scientific_codegen_log.txt','zero-shot')

# get_bleu_score('r', 4, 1, 1024, 'scientific_codegen_log.txt', 'template')
# get_bleu_score('matlab', 4, 1, 1024, 'scientific_codegen_log.txt','template')
# get_bleu_score('julia', 4, 1, 1024, 'scientific_codegen_log.txt','template')
