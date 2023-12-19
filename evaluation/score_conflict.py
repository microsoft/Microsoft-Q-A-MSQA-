from llm_eval import llm_eval_prompt, parse_score_from_review
from tqdm import tqdm
import os
import json


def tmp_conflict_stat(data_path, save_path, gptversion):
    os.makedirs(save_path, exist_ok=True)
    for filename in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, filename)
        with open(file_path) as f:
            data=json.load(f)
        question=data['QuestionText']
        grounded_answer=data['ProcessedAnswerText']
        gpt=data['GPT']
        dpr=data['GPT_DPR']
        llama=data['GPT_Llama']
        
        review_gpt_llama=llm_eval_prompt(question, grounded_answer, gpt, llama, gptversion)
        score_gpt_llama=parse_score_from_review(review_gpt_llama)
        print(review_gpt_llama)
        print(score_gpt_llama)    
        review_llama_gpt=llm_eval_prompt(question, grounded_answer, llama, gpt, gptversion)
        score_llama_gpt=parse_score_from_review(review_llama_gpt)
        print(review_llama_gpt)
        print(score_llama_gpt)
        print("=============================")
        review_gpt_dpr=llm_eval_prompt(question, grounded_answer, gpt, dpr, gptversion)
        score_gpt_dpr=parse_score_from_review(review_gpt_dpr)
        print(review_gpt_dpr)
        print(score_gpt_dpr)
        review_dpr_gpt=llm_eval_prompt(question, grounded_answer, dpr, gpt, gptversion)
        score_dpr_gpt=parse_score_from_review(review_dpr_gpt)
        print(review_dpr_gpt)
        print(score_dpr_gpt)
        print("=============================")
        review_llama_dpr=llm_eval_prompt(question, grounded_answer, llama, dpr, gptversion)
        score_llama_dpr=parse_score_from_review(review_llama_dpr)
        print(review_llama_dpr)
        print(score_llama_dpr)
        review_dpr_llama=llm_eval_prompt(question, grounded_answer, dpr, llama, gptversion)
        score_dpr_llama=parse_score_from_review(review_dpr_llama)
        print(review_dpr_llama)
        print(score_dpr_llama)

        score_data=dict(
            GPT_LLAMA = [score_gpt_llama,score_llama_gpt],
            GPT_DPR = [score_gpt_dpr,score_dpr_gpt],
            LLAMA_DPR = [score_llama_dpr,score_dpr_llama]
        )

        output_path=os.path.join(save_path, filename)

        with open(output_path, 'w') as f:
            json.dump(score_data, f, indent=4)


def conflict_stat(data_path, save_path, gptversion):
    save_path=save_path.format(gptversion=gptversion)
    os.makedirs(save_path, exist_ok=True)
    for filename in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, filename)
        output_path=os.path.join(save_path, filename)
        if os.path.exists(output_path):
            continue
        with open(file_path) as f:
            data=json.load(f)
        question=data['QuestionText']
        grounded_answer=data['ProcessedAnswerText']
        gpt=data['GPT']
        dpr=data['GPT_DPR']
        llama=data['GPT_Llama']
        
        score_gpt_llama=parse_score_from_review(llm_eval_prompt(question, grounded_answer, gpt, llama, gptversion))
        score_llama_gpt=parse_score_from_review(llm_eval_prompt(question, grounded_answer, llama, gpt, gptversion))
        score_gpt_dpr=parse_score_from_review(llm_eval_prompt(question, grounded_answer, gpt, dpr, gptversion))
        score_dpr_gpt=parse_score_from_review(llm_eval_prompt(question, grounded_answer, dpr, gpt, gptversion))
        score_llama_dpr=parse_score_from_review(llm_eval_prompt(question, grounded_answer, llama, dpr, gptversion))
        score_dpr_llama=parse_score_from_review(llm_eval_prompt(question, grounded_answer, dpr, llama, gptversion))
        
        score_data=dict(
            GPT_LLAMA = [score_gpt_llama,score_llama_gpt],
            GPT_DPR = [score_gpt_dpr,score_dpr_gpt],
            LLAMA_DPR = [score_llama_dpr,score_dpr_llama]
        )

        

        with open(output_path, 'w') as f:
            json.dump(score_data, f, indent=4)



if __name__ == '__main__':
    data_path='../evaluation/data_reformat/'
    save_path='./tmp_conflict_score_{gptversion}/'
    conflict_stat(data_path, save_path, gptversion=3.5)
    conflict_stat(data_path, save_path, gptversion=4)
    