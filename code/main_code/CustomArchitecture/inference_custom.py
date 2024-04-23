#from custom_image_question_answer import build_transformer

from custom_image_question_answer import build_transformer
#from dataset import QuestionAnswerDataset, causal_mask
#from config import get_config, get_weights_file_path, latest_weights_file_path
from config import get_config, get_weights_file_path, latest_weights_file_path
#from transformers import  BlipForQuestionAnswering
import os
import pandas as pd
#import torch
import torch
#import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
#from nltk.translate.rouge_score import rouge_n, rouge_l
#from rouge import Rouge
#from rouge import Rouge
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from sklearn.metrics import jaccard_score
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')

import logging
from dataset import QuestionAnswerDataset, causal_mask
from tokenizers import Tokenizer
from PIL import Image
from transformers import  BlipForQuestionAnswering,BlipProcessor
blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

import torchmetrics
# CUR_DIR = os.getcwd() #custom_archi
# MAIN_CODE_DIR = os.path.dirname(CUR_DIR) #main_code
# CODE_DIR = os.path.dirname(MAIN_CODE_DIR)
# PARENT_FOLDER = os.path.dirname(CODE_DIR) #main_code
# EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
# CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
# TOKENIZER_File = CONFIG_FOLDER + os.sep + 'qa_tokenizer.json'
CUR_DIR = os.getcwd() #custom_archi
MAIN_CODE_DIR = os.path.dirname(CUR_DIR) #main_code
CODE_DIR = os.path.dirname(MAIN_CODE_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR) #main_code
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
TOKENIZER_File = CONFIG_FOLDER + os.sep + 'qa_tokenizer.json'
tokenizer = Tokenizer.from_file(str(TOKENIZER_File))
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
#train_ds_raw = xdf_data[xdf_data["split"] == 'train'].copy().reset_index()#.head(100)
val_ds_raw = xdf_data[xdf_data["split"] == 'val'].copy().reset_index()


def bleu_score(y_true, y_pred):
    bleu_scores = np.zeros(len(y_true))
    for i in range(len(y_true)):
        bleu_scores[i] = nltk.translate.bleu_score.sentence_bleu([y_true[i]], y_pred[i],
                                                                 smoothing_function=SmoothingFunction().method2)

    return np.mean(bleu_scores)
def metrics_func(metrics, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    bleu, rouge, jaccard, exact match, f1 score, meteo
    list of metrics: bleu, rouge, jaccard, exact match, f1 score, meteo
    list of aggregates : avg, sum
    :return:
    '''

    def bleu_score(y_true, y_pred):
        bleu_scores = np.zeros(len(y_true))
        for i in range(len(y_true)):
            bleu_scores[i] = nltk.translate.bleu_score.sentence_bleu([y_true[i]], y_pred[i], smoothing_function=SmoothingFunction().method2)

        return np.mean(bleu_scores)

    def rouge_score(y_true, y_pred):
        rougeL_scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        for y_pred, y_true in zip(y_pred, y_true):
            scores = scorer.score(y_pred, y_true)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

        return  avg_rougeL
    def compute_meteor_score(y_true_list, y_pred_list):
        scores = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            tokens_true = word_tokenize(y_true)
            tokens_pred = word_tokenize(y_pred)
            score = meteor_score([tokens_true], tokens_pred)
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        return avg_score
    def jaccard_similarity(y_true, y_pred):
        intersection = len(set(y_true) & set(y_pred))
        union = len(set(y_true) | set(y_pred))
        return intersection / union if union else 0.0
    def exact_match(y_true, y_pred):
        num_exact_matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return num_exact_matches / len(y_true)
    def f1_score(y_true, y_pred):
        reference_tokens = set(y_true.split())
        hypothesis_tokens = set(y_pred.split())
        if len(hypothesis_tokens) == 0:
            return 0.0
        precision = len(reference_tokens.intersection(hypothesis_tokens)) / len(hypothesis_tokens)
        recall = len(reference_tokens.intersection(hypothesis_tokens)) / len(reference_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def calculate_f1_scores(y_true_list, y_pred_list):
        f1_scores = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        average_f1_score = sum(f1_scores) / len(f1_scores)
        return average_f1_score

    xcont = 1
    xsum = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'bleu':
            # f1 score average = micro
            xmet = bleu_score(y_true, y_pred)
        elif xm == 'rouge':
            # f1 score average = macro
            xmet = rouge_score(y_true, y_pred)
        elif xm == 'meteor':
            # f1 score average =
            xmet = compute_meteor_score(y_true, y_pred)
        elif xm == 'jac':
             # Cohen kappa
            xmet = jaccard_similarity(y_true, y_pred)
        elif xm == 'em':
            # Accuracy
            xmet = exact_match(y_true, y_pred)
        elif xm == 'f1':
            # Matthews
            xmet = calculate_f1_scores(y_true, y_pred)
        else:
            xmet = 0
        res_dict[xm] = xmet
        xsum = xsum + xmet
        xcont = xcont +1

    return res_dict
def greedy_decode(model, question, question_mask,pixel_values, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    encoder_output,image_embed = model.encode(question, question_mask,pixel_values=pixel_values.squeeze(1))
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(question).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)
        # calculate output
        out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask,image_embed)
        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(question).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)
def get_model(config,vocab_len):
    model = build_transformer(vocab_len, config["question_seq_len"], config['answer_seq_len'],
                              d_model=config['d_model'])
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
                                                                                                        preload) if preload else None
    #if model_filename:
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    return model

def greedy_decode(model, question, question_mask,pixel_values, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output,image_embed = model.encode(question, question_mask,pixel_values=pixel_values.squeeze(1))
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(question).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask,image_embed)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(question).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
def greedy_decode_batch(model, question, question_mask, pixel_values, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    batch_size = question.size(0)

    # Precompute the encoder output and reuse it for every step
    encoder_output, image_embed = model.encode(question, question_mask, pixel_values=pixel_values.squeeze(1))
    decoder_input = torch.empty(batch_size, 1).fill_(sos_idx).type_as(question).to(device)
    decoder_outputs = torch.empty(batch_size, 0).type_as(question).long().to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)
        #decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, 1, -1, -1)
        # calculate output
        out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask, image_embed)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_outputs = torch.cat([decoder_outputs, next_word.unsqueeze(-1)], dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=1)
        if (next_word == eos_idx).all() or decoder_input.size(1) == max_len:
            break
        # if torch.all(torch.logical_or(next_word == eos_idx, decoder_input.size(1) == max_len)):
        #     break

    return decoder_outputs
def run_validation(model, validation_ds, tokenizer, max_len, device):
    model.eval()
    count = 0

    question_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in tqdm(validation_ds, desc="Validation"):
        #for batch in validation_ds:
            count += 1
            question_input = batch["question_input"].to(device)  # (b, seq_len)
            question_mask = batch["question_mask"].to(device)  # (b, 1, 1, seq_len)
            pixel_values = batch["pixel_values"].to(device)
            model_out = greedy_decode_batch(model, question_input, question_mask,pixel_values, tokenizer, max_len, device)

            question_text_batch = batch["question_text"]
            answer_text_batch = batch["answer_text"]
            model_out_text_batch = [tokenizer.decode(output.detach().cpu().numpy()) for output in model_out]

            question_texts.extend(question_text_batch)
            expected.extend(answer_text_batch)
            predicted.extend(model_out_text_batch)


    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    #logging.info(f"Validation CER: {cer:.4f} ")

        # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    list_of_metrics = ['bleu', 'rouge', 'jac','em','f1','meteor']
    res_dict = metrics_func(list_of_metrics, y_true=expected, y_pred=predicted)
    for key, value in res_dict.items():
        print(f"{key}: {value:.2f}")

    #print(res_dict)
    #logging.info(f"Validation WER: {wer:.4f} ")
        # Compute the BLEU metric
    # metric = torchmetrics.BLEUScore()
    # bleu = metric(predicted, expected)
    # bleu = bleu_score(expected, predicted)
    # print(f"bleu score : {bleu}")
def inference_simple(config,question_text,answer_text,image_path,device):


    image = Image.open(image_path).convert('RGB')
    pixel_values = blipprocessor(image, return_tensors="pt")
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    question_tokens = tokenizer.encode(question_text).ids
    answer_tokens = tokenizer.encode(answer_text).ids

    question_num_padding_tokens = config['question_seq_len'] - len(question_tokens) - 2  # We will add <s> and </s>
    answer_num_padding_tokens = config['answer_seq_len'] - len(answer_tokens) - 1  # We will only add <s>, and </s> only on the label

    if question_num_padding_tokens < 0 or answer_num_padding_tokens < 0:
        print(question_text)
        raise ValueError("Sentence is too long")
    question_input = torch.cat(
        [
            sos_token,
            torch.tensor(question_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * question_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    # Add only <s> token
    answer_input = torch.cat(
        [
            sos_token,
            torch.tensor(answer_tokens, dtype=torch.int64),
            torch.tensor([pad_token] * answer_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )

    # Add only </s> token
    label = torch.cat(
        [
            torch.tensor(answer_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * answer_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )
    question_mask = (question_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    answer_mask = (answer_input != pad_token).unsqueeze(0).int() & causal_mask(answer_input.size(0))

    model_out = greedy_decode(model, question_input, question_mask, pixel_values, tokenizer, config['answer_seq_len'], device)
    model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())
    return model_out_text
    #encoder_output,image_embed = model.encode(question_input, question_mask,pixel_values=image_embedding.squeeze(1))
def get_ds(config):
    val_ds = QuestionAnswerDataset(val_ds_raw, tokenizer, config['answer_seq_len'], config['question_seq_len'])
    val_dataloader = DataLoader(val_ds, batch_size=20, shuffle=True)
    return val_dataloader

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    # if (device == 'cuda'):
    #     print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    #     print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    # elif (device == 'mps'):
    #     print(f"Device name: <mps>")
    # else:
    #     print("NOTE: If you have a GPU, consider using it for training.")
    #     print(
    #         "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    #     print(
    #         "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    device = torch.device(device)
    config = get_config()

    val_dataloader = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    run_validation(model, val_dataloader, tokenizer, config['answer_seq_len'], device)




