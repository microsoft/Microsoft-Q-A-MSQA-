import json
import os
import random
import pickle
# combine json files in a folder to a single jsonl file contains a list of json

path='./human_eval_data/'
combine_json=[]
sample_size=30

for filename in os.listdir(path):
    if filename.endswith('.json'):
        with open(path+filename, 'r') as f:
            data=json.load(f)
            if 'not support' in data['answer'].lower():
                continue
            combine_json.append(data)
    if len(combine_json)==sample_size:
        break
print(len(combine_json))

with open('combine.json', 'w') as f:
    json.dump(combine_json, f, indent=4)


# shuffle the question mapping
BASE_PATH = "combine.json"
SEED=45

with open(BASE_PATH, "r") as file:
    data = json.load(file)

ids=[i for i in range(len(data))]
random.seed(SEED)
random.shuffle(ids)

print('start shuffle mapping generation')

choice_mapping=[
    [1,2,3],
    [1,3,2],
    [2,1,3],
    [2,3,1],
    [3,1,2],
    [3,2,1]
]

shuffle_mapping={}

for id in ids:
    choice_id=id%6
    choice=choice_mapping[choice_id]
    shuffle_mapping[id]=choice

print('save shuffle mapping')

with open('shuffle_mapping.json', 'w') as f:
    json.dump(shuffle_mapping, f, indent=4)

print('save mapping done')