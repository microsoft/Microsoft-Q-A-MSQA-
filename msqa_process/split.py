import pandas as pd
import json

def build_json(task='train'):
    df = pd.read_csv(f"data/msqa-p-32k-{task}.csv") # isLong isShort Split
    task_name = f"MSQA_{task}"

    print(df.shape)

    save_list = []
    for row_id, row in df.iterrows():
        question, answer ,qid, tags = row.QuestionText, row.DoubleProcessedAnswerText, row.QuestionId, row.Tags
        tags = json.loads(tags.replace("'", '"'))
        
        if len(tags) == 0:
            _instruction = "Please answer the following question."
        else:
            _instruction = "Please answer the following question related to "
            for tag in tags:
                _instruction += tag+", "
            _instruction = _instruction.strip(", ")
            _instruction += '.'
        # print(_instruction)
        
        _input = f"{question}"
        _output = f"{answer}"
        d = {'instruction':_instruction,'input':_input,'output':_output,'QuestionId':qid}
        save_list.append(d)

    print(len(save_list))

    with open(f"data/{task_name}.json", 'w') as f:
        f.write(json.dumps(save_list, indent=4))


df_org = pd.read_csv("data/msqa-32k-reprocessed.csv")

df_filter = df_org[(df_org.IsAzure)&(~df_org.isShort)&(~df_org.isLong)]

df_filter['Split']=''

with open('data/test_id.txt') as f:
    test_ids = f.readlines()

test_ids=[int(id.strip()) for id in test_ids]

df_filter.loc[df_filter['QuestionId'].isin(test_ids),'Split']='Test'
df_filter.loc[~df_filter['QuestionId'].isin(test_ids),'Split']='Train'

df_test=df_filter[df_filter['Split']=='Test']
df_train=df_filter[df_filter['Split']=='Train']

df_test.to_csv("data/msqa-p-32k-test.csv", index=False)
df_train.to_csv("data/msqa-p-32k-train.csv", index=False)

print(f"test data len {len(df_test)}")
print(f"train data len {len(df_train)}")

build_json(task='train')
build_json(task='test')

