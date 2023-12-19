import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def annotation_stats(annotation_path):
    with open(annotation_path) as f:
        data=pd.read_csv(f)
    methods = ['GPT', 'GPT_DPR', 'GPT_Llama']
    # get the I don't know ratio first
    ratios = {}
    total_rows = len(data)
    for method in methods:
        don_t_know_count = data[data[method] == 3].shape[0]
        ratio = don_t_know_count / total_rows
        ratios[method] = ratio
    print('I don\'t know ratio:')
    print(ratios)

    favorite_GPT=0
    favorite_GPT_DPR=0
    favorite_GPT_Llama=0

    for index, row in data.iterrows():
        if row['GPT'] == 0:
            favorite_GPT += 1
        if row['GPT_DPR'] == 0:
            favorite_GPT_DPR += 1
        if row['GPT_Llama'] == 0:
            favorite_GPT_Llama += 1
    print(f'favor GPT: {favorite_GPT/total_rows}, favor GPT_DPR: {favorite_GPT_DPR/total_rows}, favor GPT_Llama: {favorite_GPT_Llama/total_rows}')


    for method in methods:
        # data = data[data[method] != 3]
        data[method] = data[method].replace(3, np.nan)
        data[method] = data[method].apply(lambda x: x + 1 if x in [0, 1, 2] else x)
    # print(data)
    means = data[methods].mean()
    print('means:')
    print(means)



def annotation_agreement_ratio(annotation_ratio_path):
    with open(annotation_ratio_path) as f:
        data=pd.read_csv(f)
    methods = ['GPT', 'GPT_DPR', 'GPT_Llama']
    means = data[methods].mean()
    print('agreement means:')
    print(means)

def annotation_plot(annotation_ratio_path):
    with open(annotation_ratio_path) as f:
        df=pd.read_csv(f)
    methods = ['GPT', 'GPT_DPR', 'GPT_Llama']

    thresholds = [0.5, 0.75, 1]
    data={'Methods': methods}

    for idx, threshold in enumerate(thresholds):
        new_df = df.copy()
        new_df[methods] = new_df[methods].applymap(lambda x: 1 if x >= threshold else 0)
        new_df=new_df[methods].mean()
        data['>='+str(idx+2)+' Agg']=new_df

    data = pd.DataFrame(data)
    df_melted = data.melt('Methods', var_name='Threshold', value_name='Ratio')

    # Calculate the ratio of 1's for each method for both thresholds
    fig, ax = plt.subplots(figsize=(14, 9))
    # sns.set(rc={'figure.figsize':(20,7)})

    # Plot the stacked bar plot
    sns.barplot(x='Methods', y='Ratio', hue='Threshold', data=df_melted, color=sns.color_palette("colorblind", 10), palette='bright')
    new_method_names=['LLM', 'LLM+DPR', 'LLM+EXP']
    ax.set_xticklabels(new_method_names)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    # Customize the plot
    plt.legend(loc = "upper right",fontsize=20,bbox_to_anchor=(1.05, 1))
    plt.xlabel('Method', fontsize=30)
    plt.ylabel('Percentage (%)', fontsize=30)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)

    # plt.title('Stacked Bar Plot')

    # Save the plot
    # plt.savefig(img_path)
    # Display the plot
    plt.show()
    plt.close()

        

if __name__ == '__main__':
    annotation_path='./annotation_agreement.csv'
    agreement_ratio_path='./annotation_agreement_ratio.csv'
    annotation_stats(annotation_path)
    annotation_agreement_ratio(agreement_ratio_path)
    annotation_plot(agreement_ratio_path)    