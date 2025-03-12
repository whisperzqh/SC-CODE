import nltk
from nltk.translate import meteor_score
import json

def read_jsonl(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for l in f.readlines():
        data.append(json.loads(l))
    f.close()
    return data

def calc_meteor(generated_text, reference_text):
    generated_text_list=generated_text.split()
    reference_text_list=reference_text.split()

    meteor = meteor_score.single_meteor_score(reference_text_list, generated_text_list)

    return meteor

if __name__ == '__main__':
    # for model in ['starcoder','codellama','deepseek','incoder']:
    for model in ['incoder']:
        for language in ['Julia','R','MATLAB']:
            for mode in ['template', 'retrieval']:
                shot = '1'
                references = read_jsonl(f'SC-API-CODE/NL_code/{language}.jsonl')
                data = read_jsonl(f'{model}/code_summarization/{language}_{mode}_{shot}shot.jsonl')
                scores=[]
                for i in range(len(data)):
                    reference_text = references[i]['docstring']
                    # if f'doc_{mode}{shot}' in data[i].keys():
                    #     generated_text = data[i][f'doc_{mode}{shot}'].split('\n')[0]
                    # else:
                    #     generated_text = ''
                    generated_text = data[i]['infills'][0].split('\n')[0]
                    score=calc_meteor(generated_text, reference_text)
                    scores.append(score)
                print(f'{model}_{language}_{mode}_{shot}:',round(sum(scores)/len(scores)*100,2))
        
    

