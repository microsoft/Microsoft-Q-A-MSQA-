import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter


def conflict_stat(data_path, img_path, gptversion):    
    data_path=data_path.format(gptversion=gptversion)
    conflict_list=[]
    no_conflict_list=[]
    buckets=[0,1,2,3,4,5,np.inf]
    os.makedirs(img_path, exist_ok=True)
    img_path=os.path.join(img_path,f'conflict_stat_{gptversion}.png')

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        for key in data:
            value=data[key]
            if len(value[0])==0 or len(value[1])==0:
                continue
            pos1_diff=value[0][0]-value[0][1]
            pos2_diff=value[1][0]-value[1][1]

            # no conflict
            if pos1_diff*pos2_diff<0 or (pos1_diff==0 and pos2_diff==0):
                # gap=abs(abs(pos1_diff)-abs(pos2_diff))
                # no_conflict_list.append(gap)
                no_conflict_list.append(abs(pos1_diff))
                no_conflict_list.append(abs(pos2_diff))
            # conflict
            elif pos1_diff*pos2_diff>0 or (pos1_diff==0 and pos2_diff!=0) or (pos1_diff!=0 and pos2_diff==0):
                # gap=abs(abs(pos1_diff)-abs(pos2_diff))
                # conflict_list.append(gap)
                conflict_list.append(abs(pos1_diff))
                conflict_list.append(abs(pos2_diff))
            else:
                print("error")

    conflict_counts,_=np.histogram(conflict_list,buckets)
    no_conflict_counts,_=np.histogram(no_conflict_list,buckets)

    print(len(conflict_counts),len(no_conflict_counts))

    # turns into percentage
    total_count = sum(conflict_counts) + sum(no_conflict_counts)
    print(f'the percentage of conflict is {sum(conflict_counts)/total_count}')
    print(f'the percentage of no conflict is {sum(no_conflict_counts)/total_count}')

    print('conflict vs no conflict ratio:')
    ratio = [conflict_counts[idx]/no_conflict_counts[idx] for idx in range(len(no_conflict_counts))]
    print(ratio)

    conflict_counts=conflict_counts/total_count
    no_conflict_counts=no_conflict_counts/total_count

    categories=['0','1','2','3','4','≥5']
    
    data = {'Categories': categories, 'Conflict': conflict_counts, 'No_Conflict': no_conflict_counts}
    df = pd.DataFrame(data)
    df_melted = df.melt('Categories', var_name='Values', value_name='Count')
    
    print(df_melted)

    fig, ax = plt.subplots(figsize=(12, 9))
    # sns.set(rc={'figure.figsize':(20,7)})

    # Plot the stacked bar plot
    sns.barplot(x='Categories', y='Count', hue='Values', data=df_melted, color=sns.color_palette("colorblind", 10), palette='bright')
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    # Customize the plot
    plt.legend(loc = "upper right",fontsize=30)
    plt.xlabel('Score Gap', fontsize=30)
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

    result_path = './tmp_conflict_score_{gptversion}/'
    img_path = './img/'
    conflict_stat(result_path, img_path, 3.5)
    conflict_stat(result_path, img_path, 4)
                        


