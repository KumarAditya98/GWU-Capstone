# Replicating the work of - https://medium.com/data-science-at-microsoft/visual-question-answering-with-multimodal-transformers-d4f57950c867

# Installing requirements file

# Importing required files
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
# print('Memory Usage:')
# print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
# print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Dataset pre-processing
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import argparse
import yaml
from typing import Text
import logging

def processDaquarDataset(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logging.basicConfig(level=logging.INFO)

    image_pattern = re.compile("( (in |on |of )?(the |this )?(image\d*) \?)")

    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["all_qa_pairs_file"])) as f:
        qa_data = [x.replace("\n", "") for x in f.readlines()]
    logging.info("Loaded all question-answer pairs")

    # with open("train_images_list.txt") as f:
    #     train_imgs = [x.replace("\n", "") for x in f.readlines()]

    # with open("test_images_list.txt") as f:
    #     test_imgs = [x.replace("\n", "") for x in f.readlines()]

    df = pd.DataFrame(
        {config["data"]["question_col"]: [], config["data"]["answer_col"]: [], config["data"]["image_col"]: []})

    logging.info("Processing raw QnA pairs...")
    for i in range(0, len(qa_data), 2):
        img_id = image_pattern.findall(qa_data[i])[0][3]
        question = qa_data[i].replace(image_pattern.findall(qa_data[i])[0][0], "")
        record = {
            config["data"]["question_col"]: question,
            config["data"]["answer_col"]: qa_data[i + 1],
            config["data"]["image_col"]: img_id,
        }
        df = df.append(record, ignore_index=True)

    logging.info("Creating space of all possible answers")
    answer_space = []
    for ans in df.answer.to_list():
        answer_space = answer_space + [ans] if "," not in ans else answer_space + ans.replace(" ", "").split(",")

    answer_space = list(set(answer_space))
    answer_space.sort()
    with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"]), "w") as f:
        f.writelines("\n".join(answer_space))

    # train_df = df[df.image_id.isin(train_imgs)]
    # test_df = df[df.image_id.isin(test_imgs)]

    logging.info("Splitting into train & eval sets")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]), index=None)
    test_df.to_csv(os.path.join(config["data"]["dataset_folder"], config["data"]["eval_dataset"]), index=None)
    # df.to_csv("data.csv", index=None)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    processDaquarDataset(args.config)