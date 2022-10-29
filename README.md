# dstc11-simmc2.1-damo-convai

## **Overview**

DSTC11-Track 1 : The Third Situated Interactive MultiModal Conversations (SIMMC 2.1) Challenge 2022

Team: damo-convai

For task 1, 2 and 3, we design discriminative models based on transformer-encoder structure (Longformer). To predict ambiguous candidates, coreference resolution, and belief state, we encode dialogue history and attach task-specific heads to the output of encoder. Additionally, we line up the item vectors (bbox position embedding) with their respective attribute tokens in the inputted token embedding. Auxiliary heads for predicting attributes are added as additional supervision signals.

For task 4, we propose a generative multimodal model which takes dialogue history and non-visual attributes as textual input, takes corresponding scene images as visual input and generates system response autoregressively.

## **Result**

For the results of each task, we put the prediction results of the test-std set in the corresponding folder

## **Environment**
Install the conda virtual environment by:
```shell
conda env create -f simmc.yml
cd task4
pip install -r requirements.txt
```

## **Data Preparation**

For task 1, 2 and 3, download SIMMC 2.1 data and rearrange the `data_dstc11` folder in the following format.

```
|-- images                                                # scene images
|   |-- cloth_store_1_1_1.png
|   |-- cloth_store_1_1_2.png
|   `-- ...
|-- jsons                                                 # bbox and scene jsons
|   |-- cloth_store_1_1_1_bbox.json
|   |-- cloth_store_1_1_1_scene.json
|   `-- ...
|-- fashion_prefab_metadata_all.json                      # metadata (fashion)
|-- furniture_prefab_metadata_all.json                    # metadata (furniture)
|-- simmc2.1_dials_dstc11_dev.json                          # dialogue data (dev)
|-- simmc2.1_dials_dstc11_devtest.json                      # dialogue data (devtest)
|-- simmc2.1_dials_dstc11_teststd_public.json               # dialogue data (teststd)
`-- simmc2.1_dials_dstc11_train.json                        # dialogue data (train)
```

For task 4, you can directly put SIMMC 2.1 data into the `data_dstc11` folder without rearragement.

**NOTE**: Some of the scene images are corrupted and therefore ignored. We do not make use of images in this model other than getting image size.
```
cloth_store_1416238_woman_4_8.png
cloth_store_1416238_woman_19_0.png
cloth_store_1416238_woman_20_6.png
```

## **Evaluation**

For each task, we provide the parameters of our model and the runnable code. The evaluation can be performed by running the corresponding bash file.

### **(Subtask 1) Ambiguous Candidate Identification**
```shell
cd task1/bash
bash run_dstc11_task1.sh
```

### **(Subtask 2) Multimodal Coreference Resolution**
```shell
cd task2/bash
bash run_dstc11_task2.sh
```

### **(Subtask 3) Multimodal Dialog State Tracking (MM-DST)**
```shell
cd task3/bash
bash run_dstc11_task3.sh
```

**NOTE**: For task 1, 2 and 3, the preprocessing program need to be executed in advance `taskN/scripts/process_for_dstc11_taskN.py`, and the preprocessed dataset can be found under `taskN/data` directory.
All script will print the result (Precision/Recall/F1-score) and create a line-by-line *.json prediction for each turn of the preprocessed dataset.

### **(Subtask 4) Multimodal Dialog Response Generation**
```shell
cd task4/run_scripts/simmc2.1
bash evaluate_one.sh task4_para 0 1
```

**NOTE**: For task 4, the preprocessing program need to be executed in advance `task4/dataset/gen_simmc2.1_teststd.py`, and the preprocessed tsv format dataset can be found under `task4/dataset/simmc2.1` directory.

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
