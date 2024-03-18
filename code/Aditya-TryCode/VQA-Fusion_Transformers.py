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
    #display(image)
    #img = PIL.Image.open(...)
    plt.imshow(image)
    plt.show()
    # with Image.open(os.path.join(os.getcwd(), "dataset", "images", data[id]["image_id"] + ".png")) as image:
    #     image.show()
    print("Question:\t", data[id]["question"])
    print("Answer:\t\t", data[id]["answer"], "(Label: {0})".format(data[id]["label"]))

showExample()

# Create a Multimodal Collator for the Dataset
# This will be used in the Trainer() to automatically create the Dataloader from the dataset to pass inputs to the model
#
# The collator will process the question (text) & the image, and return the tokenized text (with attention masks) along with the featurized image (basically, the pixel values). These will be fed into our multimodal transformer model for question answering.
@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join("..", "dataset", "images", image_id + ".png")).convert('RGB') for image_id
                    in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }

# Multimodal models can be of various forms to capture information from the text & image modalities, along with some cross-modal interaction as well. Here, we explore "Fusion" Models, that fuse information from the text encoder & image encoder to perform the downstream task (visual question answering).
#
# The text encoder can be a text-based transformer model (like BERT, RoBERTa, etc.) while the image encoder could be an image transformer (like ViT, Deit, BeIT, etc.). After passing the tokenized question through the text-based transformer & the image features through the image transformer, the outputs are concatenated & passed through a fully-connected network with an output having the same dimensions as the answer-space.
#
# Since we model the VQA task as a multi-class classification, it is natural to use the Cross-Entropy Loss as the loss function.

class MultimodalVQAModel(nn.Module):
    def __init__(self, pretrained_text_name, pretrained_image_name, num_labels=len(answer_space), intermediate_dim=512,dropout=0.5):
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name

        # Pretrained transformers for text & image featurization
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)

        # Fusion layer for cross-modal interaction
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Fully-connected classifier
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out

# We plan to experiment with multiple pretrained text & image encoders for our VQA Model. Thus, we will have to create the corresponding collators along with the model (tokenizers, featurizers & models need to be loaded from same pretrained checkpoints)
def createMultimodalVQACollatorAndModel(text='bert-base-uncased', image='google/vit-base-patch16-224-in21k'):
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )


    multi_model = MultimodalVQAModel(pretrained_text_name=text, pretrained_image_name=image).to(device)
    return multi_collator, multi_model

# WUPS computes the semantic similarity between two words or phrases based on their longest common subsequence in the taxonomy tree. This score works well for single-word answers (hence, we use it for our task), but may not work for phrases or sentences.
# nltk has an implementation of Wu & Palmer similarity score based on the WordNet taxanomy.
def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a)
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score

def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)

labels = np.random.randint(len(answer_space), size=5)
preds = np.random.randint(len(answer_space), size=5)

def showAnswers(ids):
    print([answer_space[id] for id in ids])

showAnswers(labels)
showAnswers(preds)

print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
print("Labels vs Labels: ", batch_wup_measure(labels, labels))

def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }
# Training code - to be looked into later for enhancements

# args = TrainingArguments(
#     output_dir="checkpoint",
#     seed=12345,
#     evaluation_strategy="steps",
#     eval_steps=100,
#     logging_strategy="steps",
#     logging_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     save_total_limit=3,        # Save only the last 3 checkpoints at any given time while training
#     metric_for_best_model='wups',
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     remove_unused_columns=False,
#     num_train_epochs=5,
#     fp16=True,
#     # warmup_ratio=0.01,
#     # learning_rate=5e-4,
#     # weight_decay=1e-4,
#     # gradient_accumulation_steps=2,
#     dataloader_num_workers=8,
#     load_best_model_at_end=True,
# )
#
# def createAndTrainModel(dataset, args, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224-in21k',
#                         multimodal_model='bert_vit'):
#     collator, model = createMultimodalVQACollatorAndModel(text_model, image_model)
#
#     multi_args = deepcopy(args)
#     multi_args.output_dir = os.path.join("..", "checkpoint", multimodal_model)
#     multi_trainer = Trainer(
#         model,
#         multi_args,
#         train_dataset=dataset['train'],
#         eval_dataset=dataset['test'],
#         data_collator=collator,
#         compute_metrics=compute_metrics
#     )
#
#     train_multi_metrics = multi_trainer.train()
#     eval_multi_metrics = multi_trainer.evaluate()
#
#     return collator, model, train_multi_metrics, eval_multi_metrics
#
# collator, model, train_multi_metrics, eval_multi_metrics = createAndTrainModel(dataset, args)

def loadAnswerSpace() -> List[str]:
    with open(os.path.join("dataset", "answer_space.txt")) as f:
        answer_space = f.read().splitlines()
    return answer_space


def tokenizeQuestion(text_encoder, question, device) -> Dict:
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder)
    encoded_text = tokenizer(
        text=[question],
        padding='longest',
        max_length=24,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    return {
        "input_ids": encoded_text['input_ids'].to(device),
        "token_type_ids": encoded_text['token_type_ids'].to(device),
        "attention_mask": encoded_text['attention_mask'].to(device),
    }


def featurizeImage(image_encoder, img_path, device) -> Dict:
    featurizer = transformers.AutoFeatureExtractor.from_pretrained(image_encoder)
    processed_images = featurizer(
        images=[Image.open(img_path).convert('RGB')],
        return_tensors="pt",
    )
    return {
        "pixel_values": processed_images['pixel_values'].to(device),
    }


question = "What is present on the hanger?"
img_path = "dataset/images/image100.png"

# Load the vocabulary of all answers
answer_space = loadAnswerSpace()

# Tokenize the question & featurize the image
question = question.lower().replace("?",
                                    "").strip()  # Remove the question mark (if present) & extra spaces before tokenizing
tokenized_question = tokenizeQuestion("bert-base-uncased", question, device)
featurized_img = featurizeImage("google/vit-base-patch16-224-in21k", img_path, device)

# Load the model checkpoint (for 5 epochs, final checkpoint should be checkpoint-1500)
model = MultimodalVQAModel(
    pretrained_text_name="roberta-base",
    pretrained_image_name="microsoft/beit-base-patch16-224-pt22k-ft22k",
    num_labels=len(answer_space),
    intermediate_dim=512
)
checkpoint = os.path.join("checkpoint", "roberta-beit","checkpoint-1500")
model.load_state_dict(torch.load(checkpoint))

model.to(device)
model.eval()

# Obtain the prediction from the model
input_ids = tokenized_question["input_ids"].to(device)
token_type_ids = tokenized_question["token_type_ids"].to(device)
attention_mask = tokenized_question["attention_mask"].to(device)
pixel_values = featurized_img["pixel_values"].to(device)
output = model(input_ids, pixel_values, attention_mask, token_type_ids)

# Obtain the answer from the answer space
preds = output["logits"].argmax(axis=-1).cpu().numpy()
answer = answer_space[preds[0]]