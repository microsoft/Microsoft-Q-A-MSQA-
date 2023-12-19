# Microsoft Q&A (MSQA)

Microsoft Q&A (MSQA) dataset is a question-answering dataset collected from the [Microsoft Q&A forum](https://learn.microsoft.com/en-us/answers/). The dataset covers a wide range of Microsoft technologies and products, including Azure, Office 365, Windows, and more. It contains 32k QA pairs, where the answer is human-generated and selected as accepted answer.


## News
- **We arxiv our [paper](https://arxiv.org/abs/2305.11541).**
- **[Paper](https://aclanthology.org/2023.emnlp-industry.29/) got accepted in EMNLP 2023 Industry Track**


## Introduction
Recent advancements in large language models (LLMs) have demonstrated their impressive performance across various natural language processing (NLP) tasks. However, when it comes to domain-specfic problems, LLMs exhibit limited performance due to their insufficient pretraining on domain knowledge. Fine-tuning and maintaining LLMs to incorporate domain-specific knowledge could be expensive for researchers. In addition, the availability of domain-specific data is often restricted and confidential, introducing risks of data leakage when using them for fine-tuning LLMs.

Existing works leverage retrieval-based methods or external modules to extract domain-specific knowledge. However, these approaches suffer from limitations such as not retrieving all the necessary context when facing complex questions. We introduce a novel model interaction paradigm that bridges domain-general and domain-specific knowledge. Our approach involves fine-tuning a small language model, e.g., LLaMA-7B using domain documentation to align it with domain-specific knowledge. Then we instruction-tune with MSQA scenario. This paradigm replaces traditional retrieval modules with the generation of domain-specific knowledge, enabling easy maintenance and privacy protection within the specific domain.

We release MSQA data and believe that this benchmarking dataset will assist the research community in evaluating their model interaction strategies in domain-specific scenarios.

### Directory Structure
The directory structure of this repository is as follows:
- `data/`: Stores the data, which is the MSQA dataset.
- `process/`: Stores the code for data cleaning.
- `viz/`: Stores the code and files related to visualization.

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
Due to the data being collected from an online Q&A forum, the content is complex and includes a large number of decorative symbols and platform-generated content, which is hard to be used for research directly. To address this issue, we conducted a deep sampling of the collected data, summarized the existing problems, identified patterns, and designed the following data filtering pipeline:
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

![visualization-demo](https://github.com/keanudicap/MSQA/blob/main/pic/demo_pic.png)

The source files for this demo can be found at [viz_demo.html](https://github.com/keanudicap/MSQA/blob/main/viz/viz_demo.html).


## License

This project is licensed under the [MIT License](LICENSE).

The datasets are licensed under  open data license, [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/).

## Get Data

If you think the release of this dataset might infringe your copyright, please inform us via the email fangkaiyang@microsoft.com for taking down the dataset.


## Paper and Citation
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