import re
from tool import *
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    
    ans_functions = [
        replace_reference_with_link,
        remove_accept_answer_line,
        remove_ref_line,
        remove_email_notification_line, 
        remove_slash_and_dash_line, # --- ///
        remove_star_symbol_line, # ***
        detect_and_remove_pic_case, # 
        detect_and_remove_thank, #
        detect_and_remove_welcome,
        detect_and_remove_hello,
        detect_and_remove_user_mentions, 
        detect_and_remove_hope,
        detect_and_remove_know,
        detect_and_remove_regards,
        remove_symbols_only_line,
        remove_space,
        detect_name_and_remove,
        remove_names_leq2words,
        remove_symbols_only_line,
        remove_multiple_n,
        remove_space,
    ]

    ques_functions = [
        detect_and_remove_pic_case,
        detect_and_remove_symbols_only_question,
        detect_and_remove_not_en_question
    ]

    data = pd.read_csv("../data/msqa-32k.csv")
    
    results = []
    for idx, row in tqdm(data.iterrows()):
        ques = row['QuestionText']
        ans = row['AnswerText']
        qid = row['QuestionId']

        processed_ques = pipeline(ques, ques_functions)
        processed_ans = pipeline(ans, ans_functions)

        if processed_ques != '1155121439' and processed_ans != '1155121439':
            res_dict = row.to_dict()
            res_dict['ProcessedAnswerText'] = processed_ans
            results.append(res_dict)

    df = pd.DataFrame(results)
    df.dropna(inplace=True)
    print(f'{df.shape[0]} samples are saved.')
    print(df.columns)
    

    df.to_csv("../data/msqa-32k.csv", index=False)
