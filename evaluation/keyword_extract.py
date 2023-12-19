from llm_components import get_oai_completion_gpt_unified
import os
import json
from tqdm import tqdm

def keyword_span_extraction(answer, gptversion):
    default_prompt='''
    As a helpful assistant, your task is to extract the keywords or important phrases from the provided text in <TEXT>. Focus on identifying significant words or phrases that are central to the topic or convey essential information. Take into account relevant context and consider both single words and multi-word expressions as potential keywords. Phrases follow the subject-verb or subject-verb-object pattern. The phrases should determine if the action is possible or not. Please provide a list of the extracted keywords or phrases, separated by a comma. Below is the text:    
    '''

    message_list = [
        {
            "role": "system",
            "content": default_prompt
        },
        {
            "role": "user",
            "content": "<TEXT>: {text}".format(text=answer)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output

if __name__ == '__main__':

    data_path='../data/human_eval_data/'
    save_path='./data/keywords_human_eval/'
    save_file_path=os.path.join(save_path, 'keywords.json')
    os.makedirs(save_path, exist_ok=True)
    keyword_set=[]
    for filename in tqdm(os.listdir(data_path)):
        file_path=os.path.join(data_path, filename)
        with open(file_path, 'r') as f:
            data=json.load(f)
        answer=data['answer']
        output=keyword_span_extraction(answer, 4)
        output=output.split(',')
        new_data=dict()
        new_data['questionId']=data['questionId']
        new_data['keywords']=output
        keyword_set.append(new_data)
    with open(save_file_path, 'w') as f:
        json.dump(keyword_set, f, indent=4)
