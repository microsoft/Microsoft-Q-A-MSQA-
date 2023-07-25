import difflib
import pandas as pd

def foo(text1, text2, save_path):
    text1 = text1.splitlines()
    text2 = text2.splitlines()
    diff = difflib.HtmlDiff(wrapcolumn=60)
    result = diff.make_file(text1, text2)
    with open(save_path,"w") as f:
        f.write(result)



if __name__ == "__main__":
    data_path = "../data/msqa-32k.csv"
    save_path = './viz_demo.html'
    sample_idx = 42

    df = pd.read_csv(data_path)
    answer_raw = df.AnswerText.iloc[sample_idx]
    answer_processed = df.ProcessedAnswerText.iloc[sample_idx]

    print(f"####### raw:\n{answer_raw}\n####### processed:\n{answer_processed}")
    foo(answer_raw, answer_processed, save_path)