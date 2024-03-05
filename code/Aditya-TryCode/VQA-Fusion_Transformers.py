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
import matplotlib.pyplot as plt
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

# Dataset pre-processing - Can employ this methodology if target is small phrase/one word answer type of problems.
# Pre-processed the dataset - split it into question - answer - image combination for train and eval.
# Extracted all the answers and created an answer space (vocabulary or classes) since this is a small dataset - only 500 classes total.
# All the questions have 1-word/phrase answer, so we consider the entire vocabulary of answers available (answer space) & treat them as labels. This converts the visual question answering into a multi-class classification problem.

dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join(os.getcwd(), "dataset", "data_train.csv"),
        "test": os.path.join(os.getcwd(), "dataset", "data_eval.csv")
    }
)

with open(os.path.join(os.getcwd(), "dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()

# For handling multiple answers for the same question - taking the top choice but we need a better methodology for this - why not make it multilabel problem?
# Also added a label column to the dataset that maps the index of the answer from the answer space created while processing the dataset.
dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answer']
        ]
    },
    batched=True
)
print(dataset)

# Looking at some question-answering pairs
from IPython.display import display
import matplotlib.pyplot as plt
import PIL

def showExample(train=True, id=None):
    if train:
        data = dataset["train"]
    else:
        data = dataset["test"]
    if id == None:
        id = np.random.randint(len(data))
    image = Image.open(os.path.join(os.getcwd(), "dataset", "images", data[id]["image_id"] + ".png"))
    display(image)
    #img = PIL.Image.open(...)
    #plt.imshow(image)
    #plt.show()
    # with Image.open(os.path.join(os.getcwd(), "dataset", "images", data[id]["image_id"] + ".png")) as image:
    #     image.show()
    print("Question:\t", data[id]["question"])
    print("Answer:\t\t", data[id]["answer"], "(Label: {0})".format(data[id]["label"]))

showExample()
