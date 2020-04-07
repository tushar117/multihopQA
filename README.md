## MultihopQA

This project is part of course "Natural Language Processing Application" from IIIT

Multi-hop QA where answering to question requires reasoning and aggregation across several paragraphs. Most of the existing QA problems are confined to finding the answers from a single paragraph (single-hop). For example, in SQuAD questions are designed to be answered given a single paragraph as the context. MultihopQA problem not only seeks to provide correct answers but also requires explanation through selection of supporting facts.

## Requirements

install required packages using anaconda
```
conda env create -n environment.yml
```

## Data Download and Preprocessing

Run the script to download the data, including HotpotQA data and GloVe embeddings, as well as spacy packages.
```
./download.sh
```

## Model Descriptions

### Baseline Model (HotPotQA)

As described in [HotPotQA paper](https://arxiv.org/pdf/1809.09600.pdf), the baseline model was implemented by augmenting the model presented [simple multiple-paragraph RC](https://www.aclweb.org/anthology/P18-1078.pdf). They have added an extra RNN layer combined with a binary classifier for classifying supporting sentences from the paragraph.

**_preprocessing_**
```
python main.py --mode prepro --data_file hotpot_train_v1.1.json --para_limit 2250 --data_split train
python main.py --mode prepro --data_file hotpot_dev_distractor_v1.json --para_limit 2250 --data_split dev
```

**_training_**
```
python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0
```

**_results_**
| setting | split | answer EM | answer F1 | sup fact EM | sup fact F1 | joint EM | joint F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| distractor | dev | 41.8 | 55.8 | 17.5 | 61.3 | 8.7 | 36.3 | 

