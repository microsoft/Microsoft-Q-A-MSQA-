import os
from tqdm import tqdm
import json
import re
from llm_components import get_oai_completion_gpt_unified
import argparse
import pandas as pd
from statistics import mean

def llm_eval_prompt(question, ans_ref, ans1, ans2, gptversion=4):
    system_prompt='''
    You are a helpful and precise assistant for checking the quality of the answer. We would like to invite you to provide feedback on the performance of two AI assistants in answering a user's question in <Question>, comparing with the <Grounded Answer> written by human. Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
    Then, output two lines indicating the scores for Assistant 1 and 2, respectively.

    Output with the following format:
    Evaluation evidence: <your evluation explanation here>
    Score of the Assistant 1: <score>
    Score of the Assistant 2: <score>
    '''
    answers_prompt='''
    <Question>: {question}
    <Grounded Answer>: {ans_ref}
    <Assistent 1's Answer>: {ans1}
    <Assistent 2's Answer>: {ans2}
    '''
    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": answers_prompt.format(question=question, ans_ref=ans_ref, ans1=ans1, ans2=ans2)
        }
    ]

    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output

def llm_eval_prompt_rank(question, ans_ref, ans1, ans2, ans3, gptversion=4):
    system_prompt='''
    You are a helpful and precise assistant for checking the quality of the answer. We would like to invite you to provide feedback on the performance of three AI assistants in answering a user's question in <Question>, comparing with the <Grounded Answer> written by human. Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
    Then, output three lines indicating the scores for Assistant 1 and 2, respectively.

    Output with the following format:
    Evaluation evidence: <your evluation explanation here>
    Score of the Assistant 1: <score>
    Score of the Assistant 2: <score>
    Score of the Assistant 3: <score>
    '''
    answers_prompt='''
    <Question>: {question}
    <Grounded Answer>: {ans_ref}
    <Assistent 1's Answer>: {ans1}
    <Assistent 2's Answer>: {ans2}
    <Assistent 3's Answer>: {ans3}
    '''
    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": answers_prompt.format(question=question, ans_ref=ans_ref, ans1=ans1, ans2=ans2, ans3=ans3)
        }
    ]

    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output

def parse_score_from_review(review):
    try:
        scores = re.findall(r"Score of(?: the)? Assistant (\d+): (\d+)", review, re.IGNORECASE)
        output = [int(score[1]) for score in scores]
        return output
    except:
        print('Failed to parse')
        print(f'Failed to parse scores from {review}')
        return [-1,-1]

def derive_rank_from_score(scores, score_gap=2):
    score_1, score_2 = scores
    if score_1 > score_2 + score_gap:
        return 'better'
    elif score_1 < score_2 - score_gap:
        return 'worse'
    else:
        return 'tie'

def check_rank_conflict(AB_rank, AC_rank, BC_rank):
    if AB_rank == 'better':
        if AC_rank == 'better':
            if BC_rank == 'better':
                return [1,2,3]
            elif BC_rank == 'worse':
                return [1,3,2]
            else:
                return [1,2,2]
        elif AC_rank == 'worse':
            if BC_rank == 'worse':
                return [2,3,1]
            else:
                # conflict
                return [-1,-1,-1]
        else:
            if BC_rank == 'better':
                return [1,2,1]
            else:
                return [-1,-1,-1]
    elif AB_rank == 'worse':
        if AC_rank == 'worse':
            if BC_rank == 'better':
                return [3,1,2]
            elif BC_rank == 'worse':
                return [3,2,1]
            else:
                return [2,1,1]
        elif AC_rank == 'better':
            if BC_rank == 'better':
                return [2,1,3]
            else:
                return [-1,-1,-1]
        else:
            if BC_rank == 'better':
                return [2,1,2]
            else:
                return [-1,-1,-1]
    else:
        if AC_rank == 'better':
            if BC_rank == 'better':
                return [1,1,2]            
            else:
                return [-1,-1,-1]
        elif AC_rank == 'worse':
            if BC_rank == 'worse':
                return [2,2,1]
            else:
                return [-1,-1,-1]
        else:
            if BC_rank == 'tie':
                return [1,1,1]
            else:
                return [-1,-1,-1]

def llm_eval_stats(score_path):
    rank_GPT=0
    rank_DPR=0
    rank_LLAMA=0
    total_count=0
    conflict_count=0
    GPT_avg_rank=[]
    DPR_avg_rank=[]
    LLAMA_avg_rank=[]

    for filename in tqdm(os.listdir(score_path)):
        file_path = os.path.join(score_path, filename)
        if not os.path.exists(file_path):            
            continue
        with open(file_path) as f:
            data = json.load(f)
        rank=data['rank']
        total_count+=1

        if rank[0]==-1:
            conflict_count+=1
            continue
        else:
            GPT_avg_rank.append(rank[0])
            DPR_avg_rank.append(rank[2])
            LLAMA_avg_rank.append(rank[1])
            # print(rank)
            if rank[0]==1:
                rank_GPT+=1
            if rank[1]==1:
                rank_LLAMA+=1
            if rank[2]==1:
                rank_DPR+=1
    conflict_rate=conflict_count/total_count
    GPT_rate=rank_GPT/total_count
    LLAMA_rate=rank_LLAMA/total_count
    DPR_rate=rank_DPR/total_count
    return conflict_rate, GPT_rate, DPR_rate, LLAMA_rate, mean(GPT_avg_rank), mean(DPR_avg_rank), mean(LLAMA_avg_rank)

def parse_score_from_review_rank(review):
    try:
        score1 = review.split("\n")[-3]
        score2 = review.split("\n")[-2]
        score3 = review.split("\n")[-1]
        score1 = score1.split(":")[-1].strip()
        score2 = score2.split(":")[-1].strip()
        score3 = score3.split(":")[-1].strip()
        return [float(score1), float(score2), float(score3)]
    except:
        print('Failed to parse')
        # print(f'Failed to parse scores from {review}')
        return [-1, -1, -1]
    

if __name__ == '__main__':

    questionID='199227'

    with open(f'../evaluation/data_0621_reformat/{questionID}.json') as f:
        data = json.load(f)
    question=data['QuestionText']
    grounded_answer=data['ProcessedAnswerText']
    gpt=data['GPT']
    dpr=data['GPT_DPR']
    llama=data['GPT_Llama']

    review1=llm_eval_prompt_rank(question, grounded_answer, gpt, llama, dpr, gptversion=4)
    print(review1)
    print("=============================")
    review2=llm_eval_prompt_rank(question, grounded_answer, llama, gpt, dpr, gptversion=4)
    print(review2)
    print("=============================")
    review3=llm_eval_prompt_rank(question, grounded_answer, dpr, gpt, llama, gptversion=4)
    print(review3)
    print("=============================")
    print(parse_score_from_review_rank(review1))
    print("=============================")
    print(parse_score_from_review_rank(review2))
    print("=============================")
    print(parse_score_from_review_rank(review3))

    