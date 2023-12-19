from rouge import Rouge
import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import torch
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_scorer
from rouge_score import rouge_scorer
import spacy
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from statistics import mean
import nltk
nltk.download('wordnet')
warnings.filterwarnings("ignore")

# n-gram metrics

def calculate_average(two_dimensional_list):
    flattened_list = [item for sublist in two_dimensional_list for item in sublist]
    average = mean(flattened_list)
    return average

def calculate_rouge_score(reference_sentence, candidate_sentence):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_sentence, candidate_sentence)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def calculate_bleu_score(reference_sentence, candidate_sentence, tokenizer):
    # Tokenize the reference and candidate sentences using the GPT-2 tokenizer
    reference_tokens = tokenizer.tokenize(reference_sentence)
    candidate_tokens = tokenizer.tokenize(candidate_sentence)
    # Compute the BLEU score using the 4-gram weights
    weights = (0.25, 0.25, 0.25, 0.25, 0.7)
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothing_function)
    return bleu_score

def calculate_meteor_score(reference_sentence, candidate_sentence):
    reference_tokens = tokenizer.tokenize(reference_sentence)
    candidate_tokens = tokenizer.tokenize(candidate_sentence)
    meteor_result = meteor_score([reference_tokens], candidate_tokens)
    return meteor_result


# semantic metrics

def calculate_bert_score(reference_sentence, candidate_sentence):
    bert_scorer=evaluate.load('bertscore', module_type='metric')
    results=bert_scorer.compute(predictions=[candidate_sentence], references=[reference_sentence], model_type='distilbert-base-uncased', lang='en')
    return results['precision'][0], results['recall'][0], results['f1'][0]

def deprecated_calculate_bert_score(reference_sentence, candidate_sentence, batch_size=16):
    '''
    If the inputs are 2 strings, the outputs are 3 numbers.
    If the inputs are 2 lists of strings, the outputs are 3 lists.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    if type(reference_sentence) is str and type(candidate_sentence) is str:
        reference_sentence = [reference_sentence]
        candidate_sentence = [candidate_sentence]
        P, R, F1 = bert_scorer(candidate_sentence, reference_sentence, lang='en', device=device, model_type='microsoft/deberta-xlarge-mnli', verbose=False)
        P, R, F1 = P.cpu().item(), R.cpu().item(), F1.cpu().item()
    else:
        P, R, F1 = bert_scorer(candidate_sentence, reference_sentence, lang='en', device=device, model_type='microsoft/deberta-xlarge-mnli', verbose=False, batch_size=batch_size)
        P, R, F1 = P.cpu().numpy(), R.cpu().numpy(), F1.cpu().numpy()        
    return P, R, F1


def calculate_similarity_score(sentence1,sentence2, tokenizer, model):
    encoded_input = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state
    # similarity_score = cosine_similarity(embeddings[0], embeddings[1])[0][0]
    similarity_score = calculate_average(cosine_similarity(embeddings[0], embeddings[1]))
    return similarity_score

def is_no_answer(response):
    NO_ANSWER_PAT=[
        'Sorry, I do not understand your question',
        'Sorry, I cannot give a confident answer',
        'I cannot confidently confirm',
        'could you please',
    ]
    for pat in NO_ANSWER_PAT:
        if pat.lower() in response.lower():
            return 1
    return 0
    # return 'could you please' in response.lower() # 'could you please provide more

def eval_pack(ground_answer, answer, tokenizer, model):
    bleu=calculate_bleu_score(ground_answer, answer, tokenizer)
    rouge1, rouge2, rougeL=calculate_rouge_score(ground_answer, answer)
    meteor=calculate_meteor_score(ground_answer, answer)
    bert=calculate_bert_score(ground_answer, answer)
    similarity=calculate_similarity_score(ground_answer, answer, tokenizer, model)
    cannot_answer=is_no_answer(answer)
    return bleu, rouge1, rouge2, rougeL, meteor, bert, similarity, cannot_answer

def eval_full(data_path, method, tokenizer, model):
    print(f'evaluating {method}...')
    bleu_list=[]
    rouge1_list=[]
    rouge2_list=[]
    rougeL_list=[]
    meteor_list=[]
    bert_p_list=[]
    bert_r_list=[]
    bert_f1_list=[]
    bleurt_list=[]
    similarity_list=[]
    spacy_similarity_list=[]
    cannot_answer_list=[]

    for idx, filename in enumerate(tqdm(os.listdir(data_path))):    
        file_path=os.path.join(data_path, filename)
        with open(file_path, 'r') as f:
            sample = json.loads(f.read())
        answer=sample['answer']
        method_answer=sample[f'{method}']
        bleu, rouge1, rouge2, rougeL, meteor, bert, bleurt, similarity, spacy_similarity, cannot_answer=eval_pack(answer, method_answer, tokenizer, model)
        cannot_answer_list.append(cannot_answer)
        if cannot_answer==1:
            continue
        bleu_list.append(bleu)
        rouge1_list.append(rouge1)
        rouge2_list.append(rouge2)
        rougeL_list.append(rougeL)
        meteor_list.append(meteor)
        bert_p_list.append(bert[0])
        bert_r_list.append(bert[1])
        bert_f1_list.append(bert[2])
        bleurt_list.append(bleurt)
        similarity_list.append(similarity)
        spacy_similarity_list.append(spacy_similarity)

    return mean(bleu_list), mean(rouge1_list), mean(rouge2_list), mean(rougeL_list), mean(meteor_list), mean(bert_p_list), mean(bert_r_list), mean(bert_f1_list), mean(similarity_list), mean(cannot_answer_list)

if __name__ == "__main__":
    data_path='./full/'
    # model_name = "bert-base-uncased"  # or choose any other transformer model
    model_name = "gpt2"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # print('raw_gpt_35')
    method_list=[
        'raw_gpt_35',
        'bm25_gpt_35',
        'dpr_gpt_35',
        'llama_512_gpt_35',
        'llama_1024_gpt_35',
        'raw_gpt_4',
        'bm25_gpt_4',
        'dpr_gpt_4',
        'llama_512_gpt_4',
        'llama_1024_gpt_4'        
    ]
    
    for method in method_list:
        mean_bleu, mean_rouge1, mean_rouge2, mean_rougeL, mean_meteor, mean_bert_p, mean_bert_r, mean_bert_f1, mean_bleurt, mean_similarity, mean_spacy_similarity, mean_cannot_answer=eval_full(data_path, method, tokenizer, model)
        output=f'{method}: bleu: {mean_bleu}, rouge1: {mean_rouge1}, rouge2: {mean_rouge2}, rougeL: {mean_rougeL}, meteor: {mean_meteor}, bert_p: {mean_bert_p}, bert_r: {mean_bert_r}, bert_f1: {mean_bert_f1}, similarity: {mean_similarity}, cannot_answer: {mean_cannot_answer}'
        print(output)
        with open('record.txt','a') as file:
            file.write(output+'\n')

    