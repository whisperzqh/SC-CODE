import json
import os
import shutil
import time
from rank_bm25 import BM25Okapi
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model, Accelerator
from accelerate.utils import get_balanced_memory




def code_summarization(language, mode, prompt_num):
    data = read_jsonl(f'../SC-API-CODE/NL_code/{language}.jsonl')
    if not os.path.exists(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl'):
        f=open(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl','w',encoding='utf-8')
        f.close()
    already_data = read_jsonl(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl')

    if mode == 'retrieval':
        corpus = [item['code'] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a',
        encoding='utf-8') as fs:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('summarization', language, mode, prompt_num, bm25, data, i)
            
            print('------------------------'+str(i)+'---------------------------')
            
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.9, top_p=0.95)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            item = {'path': data[i]['path'], 'language': language, 
                    f'doc_{mode}{str(prompt_num)}': output_text[len_example:].strip(),
                    'code': data[i]['code']}
            json.dump(item, fs)
            print(item[f'doc_{mode}{str(prompt_num)}'])
            fs.write('\n')

    return 0


def code_generation(language, mode, prompt_num):
    
    data = read_jsonl(f'../SC-API-CODE/NL_code/{language}.jsonl')
    if not os.path.exists(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl'):
        f=open(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl','w',encoding='utf-8')
        f.close()
    already_data = read_jsonl(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl')

    if mode == 'retrieval':
        corpus = [item['docstring'] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a',
            encoding='utf-8') as fw:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('generation', language, mode, prompt_num, bm25, data, i)
            
            print('------------------------'+str(i)+'---------------------------')
            # print(prompt)
            # print('---------------------')

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.9, top_p=0.95)
                
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            item = {'path': data[i]['path'], 'language': language,
                    f'code_{mode}{str(prompt_num)}': output_text[len_example + 1:].strip(),
                    'docstring': data[i]['docstring']}
            json.dump(item, fw)
            print(item[f'code_{mode}{str(prompt_num)}'])
            fw.write('\n')

    return 0


def code_completion_line(language, mode, prompt_num):
    
    data = read_jsonl(f'../SC-API-CODE/completion_data/line-level/{language}-completion-line.jsonl')
    if not os.path.exists(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl'):
        f=open(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl','w',encoding='utf-8')
        f.close()
    already_data = read_jsonl(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl')
    retrieval_data = read_jsonl(f'../SC-API-CODE/NL_code/{language}.jsonl')

    if mode == 'retrieval':
        corpus = [item['docstring'] for item in retrieval_data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a',
            encoding='utf-8') as fw:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('completion-line', language, mode, prompt_num, bm25, data, i, retrieval_data)
            
            print('------------------------'+str(i)+'---------------------------')
            # print(prompt)
            # print('---------------------')

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.9, top_p=0.95)
                
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            item = {'path': data[i]['path'], 'language': language,
                    'output': output_text[len_example:].strip(),
                    'docstring': data[i]['docstring']}
            json.dump(item, fw)
            print(item['output'])
            fw.write('\n')

    return 0


def code_completion_api(language, mode, prompt_num):
    
    data = read_jsonl(f'../SC-API-CODE/completion_data/API-level/{language}-completion-api.jsonl')
    if not os.path.exists(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl'):
        f=open(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl','w',encoding='utf-8')
        f.close()
    already_data = read_jsonl(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl')
    retrieval_data = read_jsonl(f'../SC-API-CODE/NL_code/{language}.jsonl')

    if mode == 'retrieval':
        corpus = [item['docstring'] for item in retrieval_data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a',
            encoding='utf-8') as fw:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('completion-api', language, mode, prompt_num, bm25, data, i, retrieval_data)
            
            print('------------------------'+str(i)+'---------------------------')
            # print(prompt)
            # print('---------------------')

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.9, top_p=0.95)
                
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            item = {'path': data[i]['path'], 'language': language,
                    'output': output_text[len_example:].strip(),
                    'docstring': data[i]['docstring']}
            json.dump(item, fw)
            print(item['output'])
            fw.write('\n')

    return 0


def get_prompt(task, language, mode, prompt_num, bm25, data, idx, retrieval_data=[]):
    prompt = ''
    if mode == 'template':
        number = 0
        examples = read_jsonl('../SC-API-CODE/NL_code/template_example.jsonl')
        for i in examples:
            if number == prompt_num:
                break
            if i['language'] == language:
                if language == 'MATLAB':
                    prompt += '% ' + i['docstring'] + '\n' + i['code'] + '\n\n'
                else:
                    prompt += '# ' + i['docstring'] + '\n' + i['code'] + '\n\n'
                number += 1
    elif mode == 'retrieval':
        if task == 'summarization':
            tokenized_query = data[idx]['code'].split()
        elif task == 'generation' or task == 'completion-line' or task=='completion-api':
            tokenized_query = data[idx]['docstring'].split()
        else:
            print('error')
        scores = bm25.get_scores(tokenized_query)
        prompt_idx = np.argpartition(scores, -(prompt_num + 1))[-(prompt_num + 1):]
        number = 0
        for p_idx in reversed(prompt_idx):
            if p_idx == idx:
                continue
            else:
                if number == prompt_num:
                    break
                else:
                    if language == 'MATLAB':
                        if task == 'completion-line' or task == 'completion-api':
                            prompt += '% ' + retrieval_data[p_idx]['docstring'] + '\n' + retrieval_data[p_idx]['code'] + '\n\n'
                        else:
                            prompt += '% ' + data[p_idx]['docstring'] + '\n' + data[p_idx]['code'] + '\n\n'
                    else:
                        if task == 'completion-line' or task == 'completion-api':
                            prompt += '# ' + retrieval_data[p_idx]['docstring'] + '\n' + retrieval_data[p_idx]['code'] + '\n\n'
                        else:
                            prompt += '# ' + data[p_idx]['docstring'] + '\n' + data[p_idx]['code'] + '\n\n'
                    number += 1
    len_example = len(prompt)
    if language == 'MATLAB':
        if task == 'summarization':
            prompt ='<｜fim▁begin｜>'+ prompt + '% <｜fim▁hole｜>\n' + data[idx]['code'] + '<｜fim▁end｜>'
        elif task == 'generation':
            prompt += '% ' + data[idx]['docstring'] + '\n'
        elif task == 'completion-line':
            prompt += '% ' + data[idx]['docstring'] + '\n' + data[idx]['input']
        elif task == 'completion-api':
            prompt = '<｜fim▁begin｜>'+ prompt + '% ' + data[idx]['docstring'] + '\n' + data[idx]['input_prefix'] + '<｜fim▁hole｜>' + data[idx]['input_suffix'] + '<｜fim▁end｜>'
        else:
            print('error')
    else:
        if task == 'summarization':
            prompt ='<｜fim▁begin｜>'+ prompt + '# <｜fim▁hole｜>\n' + data[idx]['code'] + '<｜fim▁end｜>'
        elif task == 'generation':
            prompt += '# ' + data[idx]['docstring'] + '\n'
        elif task == 'completion-line':
            prompt += '# ' + data[idx]['docstring'] + '\n' + data[idx]['input']
        elif task == 'completion-api':
            prompt = '<｜fim▁begin｜>'+ prompt + '# ' + data[idx]['docstring'] + '\n' + data[idx]['input_prefix'] + '<｜fim▁hole｜>' + data[idx]['input_suffix'] + '<｜fim▁end｜>'
        else:
            print('error')
    if task == 'generation':
        len_example += len('# ' + data[idx]['docstring'])
    elif task == 'summarization' or task == 'completion-line' or task=='completion-api':
        len_example = len(prompt)
    else:
        print('error')
    return prompt, len_example


def read_jsonl(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for l in f.readlines():
        data.append(json.loads(l))
    f.close()
    return data



if __name__ == '__main__':
    model_path ="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # code summarization
    for language in ['R', 'Julia', 'MATLAB']:
        mode = 'template'
        for prompt_num in [1,2,0]:
            while True:
                success = True
                try:
                    code_summarization(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item = {}
                    fw = open(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
        
        mode = 'retrieval'
        for prompt_num in [1]:
            while True:
                success = True
                try:
                    code_summarization(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item = {}
                    fw = open(f'code_summarization/{language}_{mode}_{str(prompt_num)}shot.jsonl' ,'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break

    # code generation
    for language in ['R', 'Julia', 'MATLAB']:
        mode = 'template'
        for prompt_num in [1,0]:
            while True:
                success = True
                try:
                    code_generation(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item = {}
                    fw = open(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
        
        mode = 'retrieval'
        for prompt_num in [1]:
            while True:
                success = True
                try:
                    code_generation(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item={}
                    fw = open(f'code_generation/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
                
    
    # code completion-line
    for language in ['R', 'Julia', 'MATLAB']:
        mode = 'template'
        for prompt_num in [1,0]:
            while True:
                success = True
                try:
                    code_completion_line(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item = {}
                    fw = open(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
        
        mode = 'retrieval'
        for prompt_num in [1]:
            while True:
                success = True
                try:
                    code_completion_line(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item={}
                    fw = open(f'code_completion_line/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
                
    # code completion-api
    for language in ['R', 'Julia', 'MATLAB']:
        mode = 'template'
        for prompt_num in [1,0]:
            while True:
                success = True
                try:
                    code_completion_api(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item = {}
                    fw = open(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break
        
        mode = 'retrieval'
        for prompt_num in [1]:
            while True:
                success = True
                try:
                    code_completion_api(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    item={}
                    fw = open(f'code_completion_api/{language}_{mode}_{str(prompt_num)}shot.jsonl', 'a', encoding='utf-8')
                    json.dump(item, fw)
                    fw.write('\n')
                    fw.close()
                    success = False
                if success:
                    break


    

