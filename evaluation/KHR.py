import os
import json
import re
from fuzzywuzzy import fuzz
from tqdm import tqdm
from statistics import mean

def khr_exact_match(keyword_set, question_text, response_text):
    keyword_set = [keyword.strip() for keyword in keyword_set]
    # Create a regular expression pattern by joining the keywords with the '|' (OR) operator
    pattern = re.compile(r'\b(?:%s)\b' % '|'.join(map(re.escape, keyword_set)), flags=re.IGNORECASE)

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, question_text)
    print(matches)

    # Exclude the matched keywords from the original list
    filtered_keywords = [keyword for keyword in keyword_set if keyword.lower() not in map(str.lower, matches)]

    print(filtered_keywords)

    new_pattern=re.compile(r'\b(?:%s)\b' % '|'.join(filtered_keywords), flags=re.IGNORECASE)

    matches = re.findall(new_pattern, response_text)

    matched_keywords = list(set(matches))
    print(matched_keywords)

    hit_rate=len(matched_keywords)/len(filtered_keywords)

    return hit_rate

def khr_fuzzy_match_wq(keyword_set, response_text, question_text, threshold=40):
    keyword_set = [keyword.strip().lower() for keyword in keyword_set]
    question_text=question_text.lower()
    response_text=response_text.lower()
    matched_keywords=[]
    for keyword in keyword_set:
        similarity = fuzz.partial_ratio(keyword, question_text)
        if similarity >= threshold:  # Adjust the similarity threshold as per your requirement
            matched_keywords.append(keyword)

    filtered_keywords = [keyword for keyword in keyword_set if keyword not in matched_keywords]

    answer_matched_keywords=[]
    for keyword in filtered_keywords:
        similarity = fuzz.partial_ratio(keyword, response_text)
        if similarity >= threshold:  # Adjust the similarity threshold as per your requirement
            answer_matched_keywords.append(keyword)

    # print(answer_matched_keywords)

    if len(answer_matched_keywords)==0:
        return 0

    hit_rate=len(answer_matched_keywords)/len(filtered_keywords)

    return hit_rate

def khr_fuzzy_match_woq(keyword_set, response_text, question_text, threshold=40):
    keyword_set = [keyword.strip().lower() for keyword in keyword_set]
    response_text=response_text.lower()

    answer_matched_keywords=[]
    for keyword in keyword_set:
        similarity = fuzz.partial_ratio(keyword, response_text)
        if similarity >= threshold:  # Adjust the similarity threshold as per your requirement
            answer_matched_keywords.append(keyword)

    # print(answer_matched_keywords)

    hit_rate=len(answer_matched_keywords)/len(keyword_set)

    return hit_rate

def hit_rate(keyword_path, data_path, method):
    with open(keyword_path, 'r') as f:
        keywords=json.load(f)
    raw_gpt_35_hit_rate=[]
    bm25_35_hit_rate=[]
    dpr_35_hit_rate=[]
    llama_35_hit_rate=[]
    raw_gpt_4_hit_rate=[]
    bm25_4_hit_rate=[]
    dpr_4_hit_rate=[]
    llama_4_hit_rate=[]
    def get_element_by_question_id(data, question_id):
        for item in data:
            if item['questionId'] == int(question_id):
                return item
        return None

    for filename in tqdm(os.listdir(data_path)):
        questionId=filename.split('.')[0]
        keyword_data =  get_element_by_question_id(keywords, questionId)
        keyword_set = keyword_data['keywords']
        file_path=os.path.join(data_path, filename)
        with open(file_path, 'r') as f:
            data=json.load(f)
        # print(data)
        question=data['question']
        raw_gpt_35=data['raw_gpt_35']
        bm25_35=data['bm25_gpt_35']
        dpr_35=data['dpr_gpt_35']
        llama_35=data['llama_512_gpt_35']
        raw_gpt_4=data['raw_gpt_4']
        bm25_4=data['bm25_gpt_4']
        dpr_4=data['dpr_gpt_4']
        llama_4=data['llama_512_gpt_4']
        raw_gpt_35_hit_rate.append(method(keyword_set, raw_gpt_35, question))
        bm25_35_hit_rate.append(method(keyword_set, bm25_35, question))
        dpr_35_hit_rate.append(method(keyword_set, dpr_35, question))
        llama_35_hit_rate.append(method(keyword_set, llama_35, question))
        raw_gpt_4_hit_rate.append(method(keyword_set, raw_gpt_4, question))
        bm25_4_hit_rate.append(method(keyword_set, bm25_4, question))
        dpr_4_hit_rate.append(method(keyword_set, dpr_4, question))
        llama_4_hit_rate.append(method(keyword_set, llama_4, question))
        
    return mean(raw_gpt_35_hit_rate), mean(bm25_35_hit_rate), mean(dpr_35_hit_rate), mean(llama_35_hit_rate), mean(raw_gpt_4_hit_rate), mean(bm25_4_hit_rate), mean(dpr_4_hit_rate), mean(llama_4_hit_rate)

        

if __name__ == '__main__':
    keyword_path='./data/keywords_human_eval/keywords.json'
    data_path='../data/human_eval_data/'

    mean_raw_gpt_35_hit_rate, mean_bm25_35_hit_rate, mean_dpr_35_hit_rate, mean_llama_35_hit_rate, mean_raw_gpt_4_hit_rate, mean_bm25_4_hit_rate, mean_dpr_4_hit_rate, mean_llama_4_hit_rate=hit_rate(keyword_path, data_path, khr_fuzzy_match_wq)

    print(f'raw_gpt_35_hit_rate: {mean_raw_gpt_35_hit_rate}, bm25_35_hit_rate: {mean_bm25_35_hit_rate}, dpr_35_hit_rate: {mean_dpr_35_hit_rate}, llama_35_hit_rate: {mean_llama_35_hit_rate}, raw_gpt_4_hit_rate: {mean_raw_gpt_4_hit_rate}, bm25_4_hit_rate: {mean_bm25_4_hit_rate}, dpr_4_hit_rate: {mean_dpr_4_hit_rate}, llama_4_hit_rate: {mean_llama_4_hit_rate}')

    mean_raw_gpt_35_hit_rate, mean_bm25_35_hit_rate, mean_dpr_35_hit_rate, mean_llama_35_hit_rate, mean_raw_gpt_4_hit_rate, mean_bm25_4_hit_rate, mean_dpr_4_hit_rate, mean_llama_4_hit_rate=hit_rate(keyword_path, data_path, khr_fuzzy_match_woq)

    print(f'raw_gpt_35_hit_rate: {mean_raw_gpt_35_hit_rate}, bm25_35_hit_rate: {mean_bm25_35_hit_rate}, dpr_35_hit_rate: {mean_dpr_35_hit_rate}, llama_35_hit_rate: {mean_llama_35_hit_rate}, raw_gpt_4_hit_rate: {mean_raw_gpt_4_hit_rate}, bm25_4_hit_rate: {mean_bm25_4_hit_rate}, dpr_4_hit_rate: {mean_dpr_4_hit_rate}, llama_4_hit_rate: {mean_llama_4_hit_rate}')

