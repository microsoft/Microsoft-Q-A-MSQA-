# Microsoft Q&A (MSQA)

Microsoft Q&A (MSQA) dataset is a question-answering dataset collected from the [Microsoft Q&A forum](https://learn.microsoft.com/en-us/answers/). The dataset covers a wide range of Microsoft technologies and products, including Azure, Office 365, Windows, and more. It contains 32k QA pairs, where the answer is human-generated and selected as accepted answer.


## News
- **We arxiv our [paper](https://arxiv.org/abs/2305.11541).**
- **[Paper](https://aclanthology.org/2023.emnlp-industry.29/) got accepted in EMNLP 2023 Industry Track.**


## Introduction
Recent advancements in large language models (LLMs) have demonstrated their impressive performance across various natural language processing (NLP) tasks. However, when it comes to domain-specfic problems, LLMs exhibit limited performance due to their insufficient pretraining on domain knowledge. Fine-tuning and maintaining LLMs to incorporate domain-specific knowledge could be expensive for researchers. In addition, the availability of domain-specific data is often restricted and confidential, introducing risks of data leakage when using them for fine-tuning LLMs.

Existing works leverage retrieval-based methods or external modules to extract domain-specific knowledge. However, these approaches suffer from limitations such as not retrieving all the necessary context when facing complex questions. We introduce a novel model interaction paradigm that bridges domain-general and domain-specific knowledge. Our approach involves fine-tuning a small language model, e.g., LLaMA-7B using domain documentation to align it with domain-specific knowledge. Then we instruction-tune with MSQA scenario. This paradigm replaces traditional retrieval modules with the generation of domain-specific knowledge, enabling easy maintenance and privacy protection within the specific domain.

Here is the framework of our model interaction paradigm

 ![model-interaction-framework](https://github.com/microsoft/Microsoft-Q-A-MSQA-/blob/main/pic/framework.PNG)

We release MSQA data and believe that this benchmarking dataset will assist the research community in evaluating their model interaction strategies in domain-specific scenarios.

### Directory Structure
The directory structure of this repository is as follows:
- `data/`: Stores the data, which is the MSQA dataset.
- `viz/`: Stores the code and files related to visualization.
- `pretrain_azure_doc/`: Stores the code which dumps azure documentation as the knowledge base for pretraining.
- `msqa_process/`: Stores the code for data cleaning. Note that the data in `data/` have been processed.
- `train/`: Stores the code for pretraining and finetuning.
- `inference/`: Stores the code for inference.
- `evaluation/`: Stores the code for evaluation.
- `ui/`: Stores the code for human evaluation UI.
- `human_annotation/`: Stores the code for performing analysis on human annotations.

### Statistics of MSQA

| Statistic                        | Value |
|-----------------------------|---------|
| Start Date | 2019-10-29 |
| End Date | 2023-05-25 |
| Date range             | 1304 days|
| # data | 32252 |
| # tags | 377 | 
| Avg. # questions per tag | 109.74 |
| Avg. # tags per question | 1.28 |
| Avg. #tokens per question | 276.43 | 
| Avg. #tokens per answer | 279.41 |
| Avg. #upvotes per question | 0.05 |
| Avg. #upvotes per answer | 0.28 |
| Avg. #upvotes per sample | 0.33 |

### Data Filtering
We first filtered the raw data collected by applying the following criteria:
- Discarded samples that contained attachments in the question or answer. Attachments are usually images, such as when a user takes a screenshot and asks a question. Since this work focuses mainly on text-based Q&A, samples containing attachments were discarded.
- Discarded samples without an "Accept Answer". Some questions were not answered, or had answers that were not accepted.
- Discarded samples with multi-turn discussions. Some questions contained multi-turn discussions, which were not within the scope of this work.

### Data Post-processing
The data were collected from an online Q&A forum, the content is complex and includes a large number of decorative symbols and platform-generated content, which is hard to be used for research directly. To address this issue, we conducted a deep sampling of the collected data, summarized the existing problems, identified patterns, and designed the following data filtering pipeline:
- Remove user-id.
- Standardize all links using the Markdown link reference syntax to organize them into a unified format.
- Remove platform-generated content, such as messages asking for upvotes or email notifications.
- Remove irregular decorative symbols added by users, such as asterisks for separation.
- Match various line breaks and replace consecutive multiple line breaks with a single one.
- Detect the length of questions and specifically label samples with questions exceeding 8192 tokens, as these may require special handling or truncation for current models.

### Data example
Below is an actual example from the MSQA dataset:

| Attribut | Value |
| --- | --- |
| QuestionId | 879 |
 AnswerId | 868 |
 CreationDate | 2019-11-06T02:18:47.397Z |
 Score | 1 |
 QuestionScore | 0 |
 AnswerScore | 1 |
 Tags | ["Universal Windows Platform (UWP)"] |
 IsAzure | False |
 IsM365 | False |
 IsOther | True |
 QuestionText | Can we control these three buttons:"The system Back, Close, Minimize, and Maximize buttons"<br>Based on the doc:    <br>    <br>[https://learn.microsoft.com/en-us/windows/uwp/design/shell/title-bar#how-much-to-customize-the-title-bar]()    <br>    <br>"When you opt for full customization, you are responsible for putting content in the title bar area, and you can define your own draggable region. **The system Back, Close, Minimize, and Maximize buttons are still available and handled by the system**, but elements like the app title are not. You will need to create those elements yourself as needed by your app."    <br>    <br>Does it mean that we cannot control these three buttons?     <br> |
 AnswerText | Hello,<br><br><br><br>Welcome to our Microsoft Q&A platform!<br><br><br><br>\*\>\>The system Back, Close, Minimize, and Maximize buttons are still available and \*\*handled by the system\*<br><br><br><br>Yes, as the document saied, you can put content in the title bar area, and define your own draggable region. But the Back, Close, Minimize, and Maximize buttons are still controlled by the system.<br><br><br><br>Thanks.<br><br> |
 Url | https://learn.microsoft.com/en-us/answers/questions/879/can-we-control-these-three-buttons-the-system-back.html |
 ProcessedAnswerText | \*\>\>The system Back, Close, Minimize, and Maximize buttons are still available and \*\*handled by the system\*<br>Yes, as the document saied, you can put content in the title bar area, and define your own draggable region. But the Back, Close, Minimize, and Maximize buttons are still controlled by the system. |

To facilitate manual inspection of the effectiveness of data filtering, we developed a script, `viz.py`, that visualizes the differences before and after data filtering. Below is an example of the visualization,

![visualization-demo](https://github.com/microsoft/Microsoft-Q-A-MSQA-/blob/main/pic/demo_pic.png)

The source files for this demo can be found at [viz_demo.html](https://github.com/keanudicap/MSQA/blob/main/viz/viz_demo.html).



## Model Interaction Code 

### Create virtual environment

- setup the virtual environment
    ```bash
    conda create -n msqa python=3.10
    conda activate msqa
    git clone https://github.com/ModelInteraction/MSQA.git
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    ```

### Process pretrain Azure documentation

- direct to `pretrain_azure_doc/` and run the below commandline to download the azure documentation for pretrain
    ```bash
    chmod +x clone_repos.sh
    ./clone_repos.sh
    ```
- extract and rename markdown files and save to `pretrain_azure_doc/data/`
    ```bash
    python save_azure.py
    ```
- split the markwdown files into json file limited with max token length for pretrain, save json file in to `pretrain_azure_doc/azure_json_output/`
    ```bash
    python process_azure.py
    ```

### Process MSQA data
- direct to `msqa_process/`
- post process the msqa data in `data/` collected from [Microsoft Q&A forum](https://learn.microsoft.com/en-us/answers/). Note that in the paper, we pick data labeled with `Azure` tags since we use Azure documentation for pretrain as our knowledge base. However, we release the full data for the community to use.
    ```bash
    python post_process.py
    ```
    *Note that the data in `data/` have been processed.*
- save the test sample ids *(QuestionId)* in `data/test_id.txt`, then split and save to train and test json, they should be saved to `msqa_process/data/MSQA_train.json` and `msqa_process/data/MSQA_test.json`, respectively.
    ```bash
    python split.py
    ```

### Pretrain and finetune

- direct to `train/`
- pretrain with Azure documentation following the commandline with DeepSpeed
    ```bash
    deepspeed train.py \
    --model_name_or_path {YOUR_MODEL_PATH} \
    --data_path {AZURE_JSON_PATH} \
    --output_dir {PRETRAIN_MODEL_SAVE_PATH} \
    --num_train_epochs 8 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
    ```

    where {AZURE_JSON_PATH} is the path where you save processed azure documentation json `pretrain_azure_doc/azure_json_output/`

- finetune with MSQA train data previously saved in `msqa_process/data/MSQA_train.json`
    ```bash
    deepspeed train.py \
    --model_name_or_path {PRETRAIN_MODEL_SAVE_PATH} \
    --data_path {MSQA_TRAIN_JSON_PATH} \
    --output_dir {FINETUNE_MODEL_SAVE_PATH} \
    --num_train_epochs 5 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
    ```

### Inference with finetuned model
- generate domain knowledge with our finetuned model with the commandline
    ```bash
    python inference.py \
    --base_model {FINETUNE_MODEL_SAVE_PATH} \
    --infer_ids_path {QUESTION_ID_TO_INFERENCE} \
    --save_path {RESULT_SAVE_PATH} \
    --batch_size 1 \
    --max_new_tokens 512 \
    --num_beams 4
    ```

### Result generation and evaluation
- Once the domain-specific model output its response to the question, we perform LLM generation taking either our domain knowledge or the chunks from retrieval-based methods.
- You should save your OAI key in the `keybook.py` and the endpoint function of LLM is in `llm_components.py`.
- Standard metrics, including BLEU, ROUGE-1/2/L, METEOR, BERT-Score, SIM, are defined in `eval_metrics.py`.
- Our proposed metrics
    - CAR is defined in `is_no_answer` in `eval_metrics.py`.
    - KHR is defined in `KHR.py` and keywords need to be extracted beforehand with `keyword_extract.py`.
    - LLM-based metrics is defined in `llm_eval.py`.
- `result_generation.py` contains all prompts to generate baseline results given either domain knowledge from our model or chunks from retrieval-based methods.
- `score_conflict.py` and `conflict_stat_plot.py` is to get the conflict analysis from LLM-based metric and visualization, respectively.

### Human evaluation UI
We also include the UI for human evaluators
- Direct to `ui/`
- Setup the python virtual environment
    ```bash
    conda create -n humaneval python=3.10
    conda activate humaneval
    pip install -r requirements.txt
    ```
- put the data to be evaluated in `ui/human_eval_data/`
- prepare the data
    ```bash
    python preprocess_human_eval_data.py
    ```
- Run the UI
    ```bash
    streamlit run qa_preference.py
    ```

### Human evaluation analysis
Direct to `human_annotataion/`
- put the `.csv` files of each human evaluator to `human_annotation/data/`
- process the human evaluation
    ```base
    python annotation_process.py
    ```
- output statistics and plot results in `annotation_stats.py`

## License

This project is licensed under the [MIT License](LICENSE).

The datasets are licensed under  open data license, [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/).

## Get Data

If you think the release of this dataset might infringe your copyright, please inform us via the email fangkaiyang@microsoft.com for taking down the dataset.


## Paper and Citation

[Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering](https://aclanthology.org/2023.emnlp-industry.29) (Yang et al., EMNLP 2023)

```
@inproceedings{yang-etal-2023-empower,
    title = "Empower Large Language Model to Perform Better on Industrial Domain-Specific Question Answering",
    author = "Yang, Fangkai  and
      Zhao, Pu  and
      Wang, Zezhong  and
      Wang, Lu  and
      Qiao, Bo  and
      Zhang, Jue  and
      Garg, Mohit  and
      Lin, Qingwei  and
      Rajmohan, Saravan  and
      Zhang, Dongmei",
    editor = "Wang, Mingxuan  and
      Zitouni, Imed",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-industry.29",
    doi = "10.18653/v1/2023.emnlp-industry.29",
    pages = "294--312",
    abstract = "Large Language Model (LLM) has gained popularity and achieved remarkable results in open-domain tasks, but its performance in real industrial domain-specific scenarios is average due to its lack of specific domain knowledge. This issue has attracted widespread attention, but there are few relevant benchmarks available. In this paper, we provide a benchmark Question Answering (QA) dataset named MSQA, centered around Microsoft products and IT technical problems encountered by customers. This dataset contains industry cloud-specific QA knowledge, an area not extensively covered in general LLMs, making it well-suited for evaluating methods aiming to enhance LLMs{'} domain-specific capabilities. In addition, we propose a new model interaction paradigm that can empower LLM to achieve better performance on domain-specific tasks where it is not proficient. Extensive experiments demonstrate that the approach following our method outperforms the commonly used LLM with retrieval methods. We make our source code and sample data available at: https://aka.ms/Microsoft{\_}QA.",
}

```

<!-- ## References

- [Ren et al., 2015] Shaoqing Ren, Kaiming He, Ross B. Girshick,
    and Jian Sun. Faster R-CNN: towards real-time
    object detection with region proposal networks. CoRR,
    abs/1506.01497, 2015.
- [Gilani et al., 2017] A. Gilani, S. R. Qasim, I. Malik, and
    F. Shafait. Table detection using deep learning. In Proc. of
    ICDAR 2017, volume 01, pages 771–776, Nov 2017.
- [Wu et al., 2019] Y Wu, A Kirillov, F Massa, WY Lo, R Girshick. [Detectron2](https://github.com/facebookresearch/detectron2)[J]. 2019.
- [Xie et al., 2016] Saining Xie, Ross B. Girshick, Piotr
    Doll´ar, Zhuowen Tu, and Kaiming He. Aggregated residual
    transformations for deep neural networks. CoRR,
    abs/1611.05431, 2016.
- [Klein et al., 2017] Guillaume Klein, Yoon Kim, Yuntian
    Deng, Jean Senellart, and Alexander M. Rush. Open-NMT:
    Open-source toolkit for neural machine translation.
    In Proc. of ACL, 2017.] -->