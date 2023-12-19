import streamlit as st
import csv
import json
import io
import streamlit.components.v1 as components

BASE_PATH = "combine.json"
SHUFFLE_MAPPING_PATH = "shuffle_mapping.json"
OPTIONS=['First', 'Second', 'Third', "I don't know"]
def SCROLL_TOP(index):
    return f'''
<script>
    var index = {index};
    var body = window.parent.document.querySelector(".main");
    body.scrollTop = 0;
</script>
'''


state=st.session_state
placeholder=st.empty()

def load_data():
    with open(BASE_PATH, "r") as file:
        data = json.load(file)
    with open(SHUFFLE_MAPPING_PATH, "r") as file:
        shuffle_mapping = json.load(file)    

    assert len(data)==len(shuffle_mapping)

    return data, shuffle_mapping

def save_annotations(annotations):
    csv_data = io.StringIO()
    fieldnames = ["QuestionId", "GPT", "GPT_DPR", "GPT_Llama"]
    writer = csv.DictWriter(csv_data, fieldnames=fieldnames)
    writer.writeheader()
    for question_id, annotation in annotations.items():        
        shuffle = state.shuffle_mapping[str(annotation["Question_index"])]
        answers_text=[annotation["Answer"+str(i)] for i in range(1,len(shuffle)+1)]
        shuffle_back=[shuffle.index(i)+1 for i in range(1, len(shuffle)+1)]
        answers_shuffle_back=[answers_text[i-1] for i in shuffle_back]        
        row = {
            "QuestionId": question_id,
            "GPT": answers_shuffle_back[0],
            "GPT_DPR": answers_shuffle_back[1],
            "GPT_Llama": answers_shuffle_back[2]
        }
        writer.writerow(row)    
    return csv_data

def test_print():
    print(f'value changed')

def annotate_data(question_index):
    print(f"question index {question_index}")
    cur_annotation = {}

    entry = state.data[state.question_index]
    shuffle = state.shuffle_mapping[str(state.question_index)]

    question_id = entry["questionId"]

    if question_id in state.annotations.keys():
        cur_annotation=state.annotations[question_id]        
    else:
        cur_annotation = {
            "Answer1": 0,
            "Answer2": 0,
            "Answer3": 0,
            "Question_index": question_index
        }

    question_text = entry["question"]
    grounded_answer_text=entry["answer"]
    GPT_text=entry["raw_gpt_4"]
    BM25_text=entry["dpr_gpt_4"]
    Llama_text=entry["llama_gpt_4"]

    answers_text=[GPT_text, BM25_text, Llama_text]
    answers_text=[answers_text[i-1] for i in shuffle]

    st.header("Annotation")
    st.markdown(f"**Question ID: {question_id}, Annotation ID: {question_index+1}**")
    # st.markdown(f"**Annotation ID:** {question_index+1}")
    st.markdown(
    """
    <style>
    hr {
        color: #FF0000; /* Specify your desired color code here */
        background-color: #FF0000; /* Specify the same color code here */
        height: 2px; /* Specify the height of the line */
        margin: 20px 0; /* Specify the margin around the line */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown( f"""---""")
    st.markdown( f'**Question Text:**')
    # st.markdown( f'---')
    st.markdown( f'{question_text}'.replace('\n','\n\n'))
    st.markdown( f"""---""")
    st.markdown( f'**Ground Truth Answer:**')
    # st.markdown( f'---')
    st.markdown(f'{grounded_answer_text}'.replace('\n','\n\n'))
    st.markdown( f"""---""")
    st.markdown(f"**Answer 1:**")
    st.markdown(f'{answers_text[0]}'.replace('\n','\n\n'))
    st.markdown( f"""---""")
    st.markdown(f"**Answer 2:**")
    st.markdown(f'{answers_text[1]}'.replace('\n','\n\n'))
    st.markdown( f"""---""")
    st.markdown(f"**Answer 3:**")
    st.markdown(f'{answers_text[2]}'.replace('\n','\n\n'))
    st.markdown( f"""---""")

    st.subheader("Evaluation")
    st.markdown( 'Please rank the above three answers by their similarity to the ground truth answer. You can set the same rank for two or more answers if you think they are equally similar to the ground truth answer. If the grounded answer is not available, please mark all preferences as option: I don\'t know ')

    print(question_id)

    

    answer1_ranking = st.radio(
        f"Answer1",
        key=f"Answer1_{question_index}",
        options=OPTIONS,
        # on_change=test_print,
        index=cur_annotation["Answer1"],
        horizontal=True
    )
    # print(f'correctness is {OPTIONS.index(answer1_ranking)}')
    # print(f"value of radio {type(correctness_rating)}")    
    answer2_ranking = st.radio(
        f"Answer2",
        key=f"Answer2_{question_index}",
        options=OPTIONS,
        index=cur_annotation["Answer2"],
        horizontal=True
    )
    answer3_ranking = st.radio(
        f"Answer3",
        key=f"Answer3_{question_index}",
        options=OPTIONS,
        index=cur_annotation["Answer3"],
        horizontal=True
    )

    question_id=state.data[question_index]["questionId"]
    next_annotation = {
        "Answer1": OPTIONS.index(answer1_ranking),
        "Answer2": OPTIONS.index(answer2_ranking),
        "Answer3": OPTIONS.index(answer3_ranking),  
        "Question_index": question_index 
    }
    if next_annotation != cur_annotation:
        state.annotations[question_id] = next_annotation
        st.experimental_rerun()

    state.annotations[question_id] = next_annotation

    # cur_annotation = {
    #     "Answer1": OPTIONS.index(answer1_ranking),
    #     "Answer2": OPTIONS.index(answer2_ranking),
    #     "Answer3": OPTIONS.index(answer3_ranking),  
    #     "Question_index": question_index      
    # }

    # state.annotations[question_id] = cur_annotation
    print(cur_annotation)

    # return cur_annotation

def nextprev_button_onclick(delta):
    def onclick():
        # state.annotations.update(cur_annotation)
        state.question_index += delta
        st.components.v1.html(SCROLL_TOP(state.data[question_index]["questionId"]), height=0)
    return onclick


print("reload")

if "data" not in state:
    state.data, state.shuffle_mapping =load_data()

# Initialize the annotation dictionary
if "annotations" not in state:
    state.annotations = {}

if "question_index" not in state:
    state.question_index = 0

# Perform annotation
st.header("Dataset Annotation")
st.info(f"Total Questions: {len(state.data)}")
st.info("""
1. The task is to determine the order of three 'answer text', the ranking of preference is to compare with grounded answer and rank these answers based on their similarity with the grounded answer. \n
2. You can set equal ranking preference for any two options. For example, Answer_1 > Answer_2 = Answer_3.\n
""")

annotations = state.annotations
question_index = state.question_index

# state.question_index+=1

if question_index < len(state.data):
    print(f"annotation size {len(annotations)}")    
    # if len(annotations)==0:
    annotate_data(question_index)    

    if question_index > 0:
        st.button("Previous", key="prev_button", on_click=nextprev_button_onclick(-1))

    if question_index < len(state.data) - 1:
        st.button("Next", key="next_button", on_click=nextprev_button_onclick(1))

    st.info(f"Annotated: {len(annotations)}, Remaining: {len(state.data) - len(annotations)}")

else:
    st.info("All questions annotated.")

# Save annotations and provide a download button
if len(annotations) > 0:
    print("call-2")
    st.download_button(
        "Download Annotations as CSV",
        data=save_annotations(annotations).getvalue(),
        file_name="annotations.csv"
    )

    

