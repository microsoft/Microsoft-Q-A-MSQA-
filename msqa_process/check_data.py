import pandas as pd
import json
import numpy as np
import re
from datetime import datetime
from transformers import AutoTokenizer
from tool import HTTP_PAT, PIC_PAT, REF_PAT
from collections import Counter
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def expand_regex_until_branket_match(content, pat_order=0, link_list=None):
    def construct_pat_str(pat_order):
        base_pat_str_1 = '!*\[[^\[]*?\]'
        base_pat_str_2 = '\(.*?\)'
        for _ in range(pat_order):
            base_pat_str_2 = '\([^\(\)]*?' + base_pat_str_2 + '[^\(\)]*?\)'
        return base_pat_str_1 + base_pat_str_2
    pat_str = construct_pat_str(pat_order)
    link_pat = re.compile(pat_str)
    if link_list is None:
        link_list = [x for x in link_pat.findall(content) if 'http' not in x]
    else:
        link_list.extend([x for x in link_pat.findall(content) if 'http' not in x])

    repeat = False
    res_link_list = []
    for lk in link_list:
        detect_part = lk.split('](')[-1]
        detect_part = '(' + detect_part
        counts = Counter(detect_part)
        left_c_num, right_c_num = counts['('], counts[')']
        if left_c_num != right_c_num:
            # print('FUNCTION: mismatch.')
            # print(lk)
            repeat = True
        else:
            res_link_list.append(lk)
    if repeat:
        pat_order += 1
        if pat_order > 20:
            raise Exception('Order > 20.')
        return expand_regex_until_branket_match(content, pat_order, res_link_list)
    else:
        return link_list

def plain_text_len(text):
    if type(text) != str:
        return 0, 0, True

    md_links = expand_regex_until_branket_match(text)
    for x in md_links:
        text = text.replace(x, "")

    http_links = HTTP_PAT.findall(text)
    for x in http_links:
        text = text.replace(x, "")
    
    pic_links = PIC_PAT.findall(text)
    for x in pic_links:
        text = text.replace(x, "")
    
    ref_links = REF_PAT.findall(text)
    for x in ref_links:
        text = text.replace(x, "")
    tks = tokenizer.encode(text)
    containLinks = len(md_links)>0 or len(http_links)>0 or len(pic_links)>0 or len(ref_links)>0
    return len(tks), len(text), containLinks # token len & char len & Contain links

if __name__ == '__main__':

    df = pd.read_csv("/data/msqa-32k-0616.csv")
    max_len_tks = 7500
    ans_min_len_tks = 30
    ques_min_len_tks = 10

    # 1. Date
    date_objects = [datetime.strptime(date[:10], "%Y-%m-%d") for date in df.CreationDate]
    earliest_date = min(date_objects)
    latest_date = max(date_objects)
    date_difference = (latest_date - earliest_date).days
    print(f"Data volume: {df.shape[0]}.")

    print("Earliest date:", earliest_date)
    print("Latest date:", latest_date)
    print("Difference in days:", date_difference)

    # 2. Tag
    df['Tags'] = [json.loads(x.replace("'",'"')) for x in df.Tags]
    tag_list = []
    for tag in df.Tags:
        tag_list.extend(tag)
    print(f"#unique tags: {len(set(tag_list))}")
    print(tag_list[:10])

    tag_dict = {x:0 for x in tag_list}
    for tag in df.Tags:
        for t in tag:
            tag_dict[t] += 1

    n = 0
    for i, v in enumerate(tag_dict.values()):
        n += v
    print(f"#Q per tag: {n/(i+1)}")

    lens = []
    for tag in df.Tags:
        lens.append(len(tag))
    lens = np.array(lens)
    print(f"#tag per Q: {lens.mean()}")

    q_lens, a_lens, all_lens = [], [], []
    for row_id, row in df.iterrows():
        q_tks = tokenizer.encode(row['QuestionText'])
        a_tks = tokenizer.encode(row['AnswerText'])
        q_lens.append(len(q_tks))
        a_lens.append(len(a_tks))
        all_lens.append(len(q_tks) + len(a_tks))
    df['QuestionTokenLength'] = q_lens
    df['AnswerTokenLength'] = a_lens
    df['SampleTokenLength'] = all_lens


    print(f"question len: {df['QuestionTokenLength'].mean()}")
    print(f"Answer len: {df['AnswerTokenLength'].mean()}")
    print(f"Total len: {df['SampleTokenLength'].mean()}")
    df['isLong'] = df['QuestionTokenLength'] > max_len_tks

    question_no_link_len = np.array([plain_text_len(x) for x in df.QuestionText])
    df['QuestionNoLinkTokenLen'] = question_no_link_len[:,0]
    df['QuestionNoLinkCharLen'] = question_no_link_len[:,1]
    df['QuestionContainLink'] = question_no_link_len[:,2] == 1
    df['QuestionIsShort'] = df['QuestionContainLink'] & (df['QuestionNoLinkTokenLen'] < ques_min_len_tks)
    # print(df['QuestionIsShort'].sum())


    answer_no_link_len = np.array([plain_text_len(x) for x in df.ProcessedAnswerText])
    df['ProcessedAnswerNoLinkTokenLen'] = answer_no_link_len[:,0]
    df['ProcessedAnswerNoLinkCharLen'] = answer_no_link_len[:,1]
    df['ProcessedAnswerContainLink'] = answer_no_link_len[:,2] == 1
    df['ProcessedAnswerIsShort'] = df['ProcessedAnswerContainLink'] & (df['ProcessedAnswerNoLinkTokenLen'] < ans_min_len_tks)
    # print(df['ProcessedAnswerIsShort'].sum())
    df['isShort'] = (df['ProcessedAnswerIsShort'] | df['QuestionIsShort']) 

    print(f"#too long: {df['isLong'].sum()}")
    print(f"#too short: {df['isShort'].sum()}")



    print("\n### Score")
    print(f"#zero socre: {(df.Score == 0).sum()}")
    print(f"#non-zero socre: {(df.Score != 0).sum()}")
    print(f"mean@all_socre: {df.Score.mean()}")
    print(f"mean@all_socre: {df[df.Score != 0].Score.mean()}")
    print("\n### Question Score")
    print(f"#zero socre: {(df.QuestionScore == 0).sum()}")
    print(f"#non-zero socre: {(df.QuestionScore != 0).sum()}")
    print(f"mean@all_socre: {df.QuestionScore.mean()}")
    print(f"mean@all_socre: {df[df.QuestionScore != 0].QuestionScore.mean()}")
    print("\n### Answer Score")
    print(f"#zero socre: {(df.AnswerScore == 0).sum()}")
    print(f"#non-zero socre: {(df.AnswerScore != 0).sum()}")
    print(f"mean@all_socre: {df.AnswerScore.mean()}")
    print(f"mean@all_socre: {df[df.AnswerScore != 0].AnswerScore.mean()}")

    print(f"isAzure:{df['IsAzure'].sum()}, isM365:{df['IsM365'].sum()}, isOther:{df['IsOther'].sum()}")

    df.to_csv("../MSQA/data/msqa-p-32k-short-long.csv", index=False)