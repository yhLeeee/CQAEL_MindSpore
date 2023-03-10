# CQAEL_MindSpore
MindSpore framework implementation for CQA Entity Linking (IJCAI'22 Community Question Answering Entity Linking via Leveraging Auxiliary Data)

## Requirements

- mindspore == 2.0.0a0
- mindspore-gpu == 1.10.1

## Dataset: QuoraEL

We construct a new dataset **QuoraEL**, which contains data of 504 CQA texts in total. The Wikipedia dump (July 2019 version) is used as the reference KB. Our data are in the folder data sets. CQAEL_dataset.json contains QuaraEL data mentioned above. Details of other files can be found in the codes for format conversion. Since our data set folder is too large, we release it [here](https://drive.google.com/drive/folders/1dW6iw268uDbBdi7opfwOAz_zFyq7DlrH).

### Data format

1. For each **question**, the following items are covered:`question title`, `question url`, `ID of question`, `answers`, `mentions in question title`, `topics` .

   `topics` includes `topic name`, `topic url`, `questions under this topic`

2. For each **answer**, the following items are covered:

   `answer url`, `answer id`, `upvote count`, `answer content`, `mentions in answer content`, `user name`, `user url`, `user history answers`, `user history questions`

3. For each **mention**, the following items are covered:

   `mention text`, `corresponding entity`, `candidates`, `gold entity index`

   `candidates` is a string and each candidate in `Candidates` is like: 

    `<ENTITY>\t<WIKIPEDIA_ID>\t<PRIOR_PROB>`

   The index of gold entity is '-1' if the mention cannot be linked to any candidates. There are 8030 mentions that can be linked to some candidate.

### Load data

The data set is constructed in **json** format. You can load it easily.

```python
import json
with open(PATH_OF_DATASET_FILE, 'r') as fp:
  data = json.load(fp)
```

### Files

- `candidate_ranking` folder: Codes of our model.
- `dataset` folder: our data are in the subfolder `cqa-el`. `CQAEL_dataset.json` contains **QuaraEL** data mentioned above. Details of other files can be found in the codes for format conversion.

### Quick Run

```python
python main.py	
```

For more details about the data set and the experiment settings, please refer to our paper.
