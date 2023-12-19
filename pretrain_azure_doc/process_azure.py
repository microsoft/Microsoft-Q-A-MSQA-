import os
from transformers import AutoTokenizer
import chardet
import re
from tqdm import tqdm
import time
import json
from collections import Counter

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

def get_timestamp():
    return(time.strftime('%m-%d-%H%M%S',time.localtime(time.time())))

def read_file_with_detected_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']
    return raw_data.decode(detected_encoding)

def extract_meta(md_text):
    pat_dict = dict(
        title_pat = re.compile('title: (.*?)\n'),
        titleSuffix_pat = re.compile('titleSuffix: (.*?)\n'),
        description_pat = re.compile('description: (.*?)\n'),
        author_pat = re.compile('author: (.*?)\n'),
        manager_pat = re.compile('manager: (.*?)\n'),
        services_pat = re.compile('services: (.*?)\n'),
        ms_author_pat = re.compile('ms.author: (.*?)\n'),
        ms_date_pat = re.compile('ms.date: (.*?)\n'),
        ms_topic_pat = re.compile('ms.topic: (.*?)\n'),
        ms_service_pat = re.compile('ms.service: (.*?)\n')
    )
    results = {}
    for meta_name, meta_pat in pat_dict.items():
        res = meta_pat.findall(md_text)
        if len(res) > 0:
            res = res[0]
        else:
            res = ''
        results[meta_name] = res
    return results


def extract_meta_block(md_text):
    parts = md_text.split('---\n')
    if len(parts) > 2:
        meta = parts[1]
        other = parts[2:]
        if type(other) is list:
            other = '\n'.join(other)
    else:
        meta = ''
        other = md_text
    return meta, other

def remove_multiple_n(text):
    text = text.replace("\r", "\n")
    return re.sub('\n+', '\n', text)

def remove_include(text):
    text = text.replace('\r', '\n')
    res = []
    for p in text.split("\n"):
        if '!INCLUDE' in p:continue
        else:res.append(p)
    return '\n'.join(res)

def split_plain_text(text, tokenizer, max_tokens):
    parts = [f'{x}.' for x in text.split('.')]

    tokens = []
    current_chunk = []
    for part in parts:
        tokenized_part = tokenizer(part)["input_ids"]
        if len(current_chunk) + len(tokenized_part) > max_tokens:
            if current_chunk:
                tokens.append(current_chunk)
            current_chunk = tokenized_part
        else:
            current_chunk.extend(tokenized_part)

    if current_chunk:
        tokens.append(current_chunk)
    return tokens

if __name__ == '__main__':
    max_seq_len = 1024
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    quotation_pat = re.compile('\[[^\[]*?\]')

    tripleColon_pat = re.compile(':::.*?:::')
    md_pat = re.compile('\[.*?\].*?\.md#?.*')
    dir_path = './data/'
    save_path = './azure_json_output/'
    os.makedirs(save_path, exist_ok=True)
    subfolders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    res = []
    for subfolder in subfolders:
        subfolder=os.path.join(dir_path, subfolder)
        print(subfolder)
        files = os.listdir(subfolder)
        for i, file in enumerate(tqdm(files)):
            file_path = os.path.join(subfolder, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
            except:
                print(file_path)
                md_text = read_file_with_detected_encoding(file_path)
            meta_block, content = extract_meta_block(md_text)
            content = remove_include(content)


            colon_3 = tripleColon_pat.findall(content)
            if len(colon_3) > 0:
                for _ in colon_3:
                    content = content.replace(_, '')

            relative_link = expand_regex_until_branket_match(content)
            for lk in relative_link:
                if '.png' in lk:
                    content = content.replace(lk, '')
                else:
                    lk_text = quotation_pat.findall(lk)[0]

                    content = content.replace(lk, lk_text)

            extra_link = md_pat.findall(content)
            extra_link = [x for x in extra_link if 'http' not in x]
            if len(extra_link) > 0:
                for lk in extra_link:
                    # print(lk)
                    content = content.replace(lk, '')
            tokens = split_plain_text(content, tokenizer, max_tokens=max_seq_len)
            res.extend(tokens)

    print(f"all: {len(res)}\nPost check...")

    final_res=  []
    too_long = too_short = 0
    for i, x in enumerate(tqdm(res)):
        text = tokenizer.decode(x)
        if len(x) > max_seq_len: too_long += 1
        elif len(x) < 30: too_short += 1
        else:
            final_res.append({"instruction": "", "input": "", "output": text} )
    print(f"all: {len(final_res)}, short: {too_short}, long: {too_long}.")

    with open(save_path+f"{get_timestamp()}.json", "w") as f:
        json.dump(final_res,f,indent=4)
        