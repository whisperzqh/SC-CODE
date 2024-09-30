import json
import time
import openai
import os
from text_generation import Client

# add secret key
chat_gpt_key = ""
openai.api_key = chat_gpt_key

HF_TOKEN = os.environ.get("HF_TOKEN", '')


def call_gpt(prompt, model):
    response = None
    while True:
        success = True
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=0.8
            )
        except Exception as e:
            print(e)
            print("fail to call gpt, trying again...")
            time.sleep(2)
            success = False
        if success:
            break

    result = response['choices'][0]['message']['content']
    return result


def call_starcoder(prompt):
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_INDICATOR = "<FILL_HERE>"
    API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"

    def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0):
        temperature = float(temperature)
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

        client = Client(API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, )
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

    output = generate(prompt=prompt+"<FILL_HERE>", temperature=0.9, max_new_tokens=192, top_p=0.95, repetition_penalty=1.0)
    for value in output:
        pass
    return value[len(prompt)+1:].strip()


def call_codellama(prompt):
    FIM_PREFIX = "<PRE> "
    FIM_MIDDLE = " <MID>"
    FIM_SUFFIX = " <SUF>"
    FIM_INDICATOR = "<FILL_ME>"
    EOS_STRING = "</s>"
    EOT_STRING = "<EOT>"
    API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-13b-hf"

    def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0, ):
        temperature = float(temperature)
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

        client = Client(API_URL,headers={"Authorization": f"Bearer {HF_TOKEN}"},)
        stream = client.generate_stream(prompt, **generate_kwargs)

        if fim_mode:
            output = prefix
        else:
            output = prompt

        previous_token = ""
        for response in stream:
            if any([end_token in response.token.text for end_token in [EOS_STRING, EOT_STRING]]):
                if fim_mode:
                    output += suffix
                    yield output
                    return output
                    # print("output", output)
                else:
                    return output
            else:
                output += response.token.text
            previous_token = response.token.text
            yield output
        return output

    output = generate(prompt=prompt+"<FILL_ME>", temperature=0.9, max_new_tokens=192, top_p=0.95, repetition_penalty=1.0)
    for value in output:
        pass
    return value[len(prompt)+1:].strip()


def read_jsonl(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for l in f.readlines():
        data.append(json.loads(l))
    f.close()
    return data


if __name__ == '__main__':
    # replace the value of 'model'
    model = 'gpt3.5'
    for language in ['Julia', 'MATLAB', 'R']:
        data = read_jsonl('../data_for_semantic_correctness/testcases_data' + language + '.jsonl')

        with open('test_result/'+model+'/' + language + '.jsonl', 'a', encoding='utf-8') as f:
            for i in range(len(data)):
                if model == ('gpt4'):
                    output = call_gpt([{'role': 'user', 'content': data[i]['prompt']}], 'gpt-4-1106-preview')
                elif model == ('gpt3.5'):
                    output = call_gpt([{'role': 'user', 'content': data[i]['prompt']}], 'gpt-3.5-turbo-1106')
                elif model == 'starcoder':
                    output = call_starcoder(data[i]['prompt'])
                elif model == 'codellama':
                    output = call_codellama(data[i]['prompt'])
                data[i]['result'] = output
                json.dump(data[i], f, ensure_ascii=False)
                f.write('\n')
                print(data[i])
