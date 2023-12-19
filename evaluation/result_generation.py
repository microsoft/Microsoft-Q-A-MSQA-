from llm_components import get_oai_completion_gpt_unified
import time
import json
import os
from tqdm import tqdm
import argparse
import random
import pandas as pd
# random.seed(2023)


# default prompts
def raw_gpt_result(question, gptversion):
    default_msqa_prompt='''
            As a helpful assistant, your task is to create responses to the user's questions. If you cannot be sure about the user's intention, please say \"Sorry, I do not understand your question\"; If you cannot give a confident answer, please say \"Sorry, I cannot give a confident answer\".
            '''
    message_list = [
        {
            "role": "system",
            "content": default_msqa_prompt
        },
        {
            "role": "user",
            "content": "{question}".format(question=question)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output

# retrieval results
def retrieval_gpt_result(question, chunks, gptversion):
    chunk_prompt = '''     
            As a helpful assistant, your task is to create responses to the user's questions. We have retrieved some chunks from documents. These chunks are incomplete paragraphs and may not be relevant to the question. Please first determine whether these chunks are related to the user's question and disregard those you deem irrelevant. For the helpful chunks, integrate the useful content from these chunks into your answer without quoting them. If you cannot be sure about the user's intention, please say \"Sorry, I do not understand your question\"; If you cannot give a confident answer, please say \"Sorry, I cannot give a confident answer\". Below are the chunks: \n\n
    '''
    # print(chunks)
    for chunk in chunks:
        chunk_prompt += '<CHUNK>\n\n' + chunk + '\n\n'
    message_list = [
        {
            "role": "system",
            "content": chunk_prompt
        },
        {
            "role": "user",
            "content": "{question}".format(question=question)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output

# llama results
def llama_gpt_result(question, llama, gptversion):
    llama_prompt = '''     
            As a helpful assistant, your task is to create responses to the user's questions. We have retrieved one response from another LLM. This answer may not be relevant to the question. If you think the LLM response is helpful, integrate the useful information into your answer without quoting them, otherwise, you can ignore the LLM response. If you cannot be sure about the user's intention, please say \"Sorry, I do not understand your question\"; If you cannot give a confident answer, please say \"Sorry, I cannot give a confident answer\". Below are the LLM response: \n\n
    '''
    # print(chunks)

    llama_prompt += '<LLM RESPONSE>\n\n' + llama + '\n\n'
    message_list = [
        {
            "role": "system",
            "content": llama_prompt
        },
        {
            "role": "user",
            "content": "{question}".format(question=question)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, gptversion)
    return output


if __name__ == '__main__':

    questionID='209355'
    # load json file
    # navigate folder in another folder
    with open(f'../evaluation/data_0621_reformat/{questionID}.json') as f:
        data = json.load(f)
    question=data['QuestionText']
    chunks=data['DPR_Chunks']
    llama=data['Llama_Chunks']

    output=raw_gpt_result(question, 3.5)
    print("gpt 3.5 ouput:")
    print(output)
    print("gpt 4 ouput:")
    output = raw_gpt_result(question, 4)
    print(output)
    print("===============================")
    output=retrieval_gpt_result(question, chunks, 3.5)
    print("gpt 3.5 chunks ouput:")
    print(output)
    print("gpt 4 chunks ouput:")
    output = retrieval_gpt_result(question, chunks, 4)
    print(output)
    print("===============================")
    output=llama_gpt_result(question, llama, 3.5)
    print("gpt 3.5 llama ouput:")
    print(output)
    print("gpt 4 llama ouput:")
    output = llama_gpt_result(question, llama, 4)
    print(output)
    
