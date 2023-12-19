import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import mode
import numpy as np

# check the annotation agreement
def annotation_agreement(annotation_df_list):
    # df_merged=annotation_df_list[0]
    # df_merged = annotation_df_list[0].add_suffix('_1')
    methods = ['GPT', 'GPT_DPR', 'GPT_Llama']
    for idx, df in enumerate(annotation_df_list):
        annotation_df_list[idx].rename(columns={method: method + '_'+str(idx+1) for method in methods}, inplace=True)

    df_merged = annotation_df_list[0]
    for idx, df in enumerate(annotation_df_list[1:], start=2):
        df_merged = pd.merge(df_merged, df, on='QuestionId')
    num_annotators = len(annotation_df_list)

    def check_agreement(row):
        mode_values=[]
        for method in methods:
            values=[row[method+'_'+str(i)] for i in range(1,num_annotators+1)]
            try:
                # print(mode(values, keepdims=True))
                mode_all=mode(values, keepdims=True)
                mode_val=mode_all[0][0][0]
                mode_values.append(mode_val)
            except:
                mode_values.append(np.nan)

        return pd.Series(mode_values, index=methods)

    def check_agreement_ratio(row):
        agreement_ratios=[]
        for method in methods:
            values=[row[method+'_'+str(i)] for i in range(1,num_annotators+1)]
            try:
                # print(mode(values, keepdims=True))
                mode_all=mode(values, keepdims=True)
                agreement_ratio=mode_all[1][0][0]/num_annotators
                agreement_ratios.append(agreement_ratio)
            except:
                agreement_ratios.append(np.nan)

        return pd.Series(agreement_ratios, index=methods)

    df_modes= df_merged.groupby('QuestionId').apply(check_agreement).reset_index()
    df_ratios= df_merged.groupby('QuestionId').apply(check_agreement_ratio).reset_index()
    return df_modes, df_ratios

if __name__ == '__main__':

    annotation_path='./data/'
    annotation_df_list=[]
    for filename in os.listdir(annotation_path):
        file_path = os.path.join(annotation_path, filename)
        with open(file_path) as f:
            data=pd.read_csv(f)
        annotation_df_list.append(data)
    df_modes, df_ratio=annotation_agreement(annotation_df_list)
    df_modes.to_csv('./annotation_agreement.csv', index=False)
    df_ratio.to_csv('./annotation_agreement_ratio.csv', index=False)