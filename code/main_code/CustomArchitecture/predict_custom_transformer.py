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
from datetime import datetime
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
TOKENIZER_File = CONFIG_FOLDER + os.sep + 'custom_tokenizer.json'
tokenizer = Tokenizer.from_file(str(TOKENIZER_File))
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)


test_data_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"

generated_result_folder = EXCEL_FOLDER  + os.sep + 'generated_result'
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not os.path.exists(generated_result_folder):
    os.mkdir(generated_result_folder)
generated_result_excel_file = f"{generated_result_folder}{os.sep}test_data_{current_time}.xlsx"
#generated_result_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"
xdf_dset_test = pd.read_excel(test_data_excel_file)#.head(10)

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


def run_validation(model, test_ds, tokenizer, config, device):
    model.eval()
    question_texts = []
    expected = []
    predicted = []
    batch_size = config['batch_size']
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    val_epoch_loss = 0.0
    val_epoch_step_size = 0
    with torch.no_grad():
        for batch in tqdm(test_ds, desc="Validation"):
            question_input = batch["question_input"].to(device)  # (b, seq_len)
            question_mask = batch["question_mask"].to(device)  # (b, 1, 1, seq_len)
            answer_input = batch['answer_input'].to(device)
            answer_mask = batch['answer_mask'].to(device)
            pixel_values = batch["pixel_values"].to(device)
            question_text = batch["question_text"]  # .to(device)
            answer_text = batch["answer_text"]  # .to(device)

            encoder_output, image_embed = model.encode(question_input, question_mask,
                                                       pixel_values=pixel_values.squeeze(1))
            ###################### decoder #########################################

            decoder_input = torch.empty(batch_size, 1).fill_(sos_idx).type_as(question_input).to(device)  # (b,1) sos
            decoder_outputs = torch.empty(batch_size, 0).type_as(question_input).long().to(device)  # (batch_size, 0)

            while True:
                if decoder_input.size(1) == config['answer_seq_len']:
                    break

                # build mask for target
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)
                decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, 1, -1, -1)
                out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask,
                                   image_embed)  # next token prob

                # get next token
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_outputs = torch.cat([decoder_outputs, next_word.unsqueeze(-1)], dim=1)
                decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=1)
                if (next_word == eos_idx).all() or decoder_input.size(1) == config[
                    'answer_seq_len']:  # all have eos or max seq len reached
                    break
            model_out = decoder_outputs

            model_out_text_batch = [tokenizer.decode(output.detach().cpu().numpy()) for output in model_out]
            question_texts.extend(question_text)
            expected.extend(answer_text)
            predicted.extend(model_out_text_batch)

    ############ Evaluation metric ######
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    #logging.info(f"Validation CER: {cer:.4f} ")
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    #logging.info(f"Validation WER: {wer:.4f} ")

    list_of_metrics = ['bleu', 'rouge', 'jac', 'em', 'f1', 'meteor']
    res_dict = metrics_func(list_of_metrics, y_true=expected, y_pred=predicted)
    res_str = ""
    for key, value in res_dict.items():
        res_str = res_str + "\n" + key + " " + str(round(value, 2))
        #logging.info(f"Validation {key}: {value:.4f} ")
    print(res_str)

def get_ds(config):
    test_ds = QuestionAnswerDataset(xdf_dset_test, tokenizer, config['answer_seq_len'], config['question_seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=20, shuffle=True)
    return test_dataloader

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    device = torch.device(device)
    config = get_config()

    test_dataloader = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    run_validation(model, test_dataloader, tokenizer, config, device)





