import json
import os
import shutil
import time

import numpy as np
import requests

import gradio as gr
import text_generation.errors
from huggingface_hub import Repository
from text_generation import Client
from rank_bm25 import BM25Okapi

from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css

# 需要补全自己的huggingface token
HF_TOKEN = os.environ.get("HF_TOKEN", '')

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

FIM_INDICATOR = "<FILL_HERE>"

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)


def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0, version="StarCoder"):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    fim_mode = False

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    if FIM_INDICATOR in prompt:
        fim_mode = True
        try:
            prefix, suffix = prompt.split(FIM_INDICATOR)
        except:
            raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt!")
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

    if version == "StarCoder":
        stream = client.generate_stream(prompt, **generate_kwargs)

    if fim_mode:
        output = prefix
    else:
        output = prompt

    previous_token = ""
    for response in stream:
        if response.token.text == "<|endoftext|>":
            if fim_mode:
                output += suffix
            else:
                return output
        else:
            output += response.token.text
        previous_token = response.token.text
        yield output
    return output


def code_summarization(language, mode, prompt_num):
    data = read_jsonl('../../SC-API-CODE/NL-code/' + language + '.jsonl')
    already_data = read_jsonl(language + '/starcoder_output_' + mode + str(prompt_num) + '.jsonl')

    if mode == 'retrieval':
        corpus = [item['code'] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(language + '/starcoder_output_' + mode + str(prompt_num) + '.jsonl', 'a',
            encoding='utf-8') as fw:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('summarization', language, mode, prompt_num, bm25, data, i)
            output = generate(prompt=prompt, temperature=0.9, max_new_tokens=32, top_p=0.95, repetition_penalty=1.0,
                              version="StarCoder")
            for value in output:
                pass
            item = {'path': data[i]['path'], 'language': language, 'code': data[i]['code'],
                    'doc_' + mode + str(prompt_num): value[len_example + 1:].strip()}
            json.dump(item, fw)
            print(str(i) + ' ' + item['doc_' + mode + str(prompt_num)])
            fw.write('\n')

    return 0


def code_generation(language, mode, prompt_num):
    data = read_jsonl('../../SC-API-CODE/NL-code/' + language + '.jsonl')
    already_data = read_jsonl(language + '/starcoder_output_' + mode + str(prompt_num) + '.jsonl')

    if mode == 'retrieval':
        corpus = [item['docstring'] for item in data]
        tokenized_corpus = [item.split() for item in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        bm25 = ''

    with open(language + '/starcoder_output_' + mode + str(prompt_num) + '.jsonl', 'a',
            encoding='utf-8') as fw:

        for i in range(len(already_data), len(data)):
            prompt, len_example = get_prompt('generation', language, mode, prompt_num, bm25, data, i)
            output = generate(prompt=prompt, temperature=0.9, max_new_tokens=192, top_p=0.95, repetition_penalty=1.0, version="StarCoder")
            for value in output:
                pass
            item = {'path': data[i]['path'], 'language': language,
                    'code_' + mode + str(prompt_num): value[len_example + 1:].strip(),
                    'docstring': data[i]['docstring']}
            json.dump(item, fw)
            print(str(i) + ' ' + item['code_' + mode + str(prompt_num)])
            fw.write('\n')

    return 0


def get_prompt(task, language, mode, prompt_num, bm25, data, idx):
    prompt = ''
    if mode == 'template':
        number = 0
        examples = read_jsonl('template_example.jsonl')
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
        elif task == 'generation':
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
                        prompt += '% ' + data[p_idx]['docstring'] + '\n' + data[p_idx]['code'] + '\n\n'
                    else:
                        prompt += '# ' + data[p_idx]['docstring'] + '\n' + data[p_idx]['code'] + '\n\n'
                    number += 1
    len_example = len(prompt)
    if language == 'MATLAB':
        if task == 'summarization':
            prompt += '% <FILL_HERE>\n' + data[idx]['code']
        elif task == 'generation':
            prompt += '% ' + data[idx]['docstring'] + '\n<FILL_HERE>'
        else:
            print('error')
    else:
        if task == 'summarization':
            prompt += '# <FILL_HERE>\n' + data[idx]['code']
        elif task == 'generation':
            prompt += '# ' + data[idx]['docstring'] + '\n<FILL_HERE>'
        else:
            print('error')
    if task == 'generation':
        len_example += len('# ' + data[idx]['docstring'])
    return prompt, len_example


def read_jsonl(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for l in f.readlines():
        data.append(json.loads(l))
    f.close()
    return data


if __name__ == '__main__':
    for language in ['R', 'Julia', 'MATLAB']:
        mode = 'retrieval'
        for prompt_num in [1, 2]:
            while True:
                success = True
                try:
                    code_generation(language, mode, prompt_num)
                except Exception as e:
                    print(e)
                    if type(e) is text_generation.errors.ValidationError:
                        item = {}
                        fw = open(language + '/starcoder_output_' + mode + str(prompt_num) + '.jsonl', 'a', encoding='utf-8')
                        json.dump(item, fw)
                        fw.write('\n')
                        fw.close()
                    time.sleep(2)
                    success = False
                if success:
                    break