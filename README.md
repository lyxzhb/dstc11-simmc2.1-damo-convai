# dstc11-simmc2.1-damo-convai

## **Overview**

DSTC11-Track 1 : The Third Situated Interactive MultiModal Conversations (SIMMC 2.1) Challenge 2022

Team: damo-convai

## **Environment**

## **Result**

For the results of each task, we put the prediction results of the test-std set in the corresponding folder

## **Training**

## **Evaluation**

## **Model Parameter**

Since our model is trained separately for each task, Download the model parameters by one of the following methods:

| Sub-Task #1 | Ambiguous Candidate Identification (New) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | Given ambiguous object mentions, to resolve referent objects to thier canonical ID(s). |
| Input | Current user utterance, Dialog context, Multimodal context |
| Output |  Canonical object IDs |
| Metrics | Object Identification F1 |
| Performance (devtest) | 70.31 |
| Checkpoint | [Checkpoint Link](task1) |

| Sub-Task #2 | Multimodal Coreference Resolution |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To resolve referent objects to thier canonical ID(s) as defined by the catalog. |
| Input | Current user utterance, Dialog context, Multimodal context |
| Output |  Canonical object IDs |
| Metrics |  Coref F1 |
| Performance (devtest) | 94.40 |
| Checkpoint | [Checkpoint Link](task2) |

| Sub-Task #3 | Multimodal Dialog State Tracking (MM-DST) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To track user belief states across multiple turns |
| Input | Current user utterance, Dialogue context, Multimodal context |
| Output | Belief state for current user utterance |
| Metrics | Slot F1, Intent F1 |
| Performance (devtest) | 94.37/99.19 |
| Checkpoint | [Checkpoint Link](task3) |

| Sub-Task #4 | Multimodal Dialog Response Generation  |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To generate Assistant responses  |
| Input | Current user utterance, Dialog context, Multimodal context, (Ground-truth API Calls) |
| Output | Assistant response utterance |
| Metrics | BLEU-4 |
| Performance (devtest) | 45.39 |
| Checkpoint | [Checkpoint Link](task4) |


## **References**

```
@inproceedings{kottur-etal-2021-simmc,
    title = "{SIMMC} 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations",
    author = "Kottur, Satwik  and
      Moon, Seungwhan  and
      Geramifard, Alborz  and
      Damavandi, Babak",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.401",
    doi = "10.18653/v1/2021.emnlp-main.401",
    pages = "4903--4912",
}
```

## **License**

Our repository is released under MIT License, see [LICENSE](LICENSE) for details.