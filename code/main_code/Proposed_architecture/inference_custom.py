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
df_to_test = xdf_dset_test.copy()
ids_to_test = [2,7,8,5]

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
    pixel_values = pixel_values.to(device)
    question = question.to(device)
    question_mask = question_mask.to(device)
    encoder_output,image_embed = model.encode(question, question_mask,pixel_values=pixel_values)
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
def inference_simple(config,question_text,answer_text,image_path,device,model):


    image = Image.open(image_path).convert('RGB')
    pixel_values = blipprocessor(image, return_tensors="pt")['pixel_values']
    #model = get_model(config, tokenizer.get_vocab_size()).to(device)
    model.eval()
    sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    with torch.no_grad():
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
def get_ds(config):
    test_ds = QuestionAnswerDataset(xdf_dset_test, tokenizer, config['answer_seq_len'], config['question_seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=True)
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

    #ids_to_test = [1,2,3,5]
    image_names = []
    image_questions = []
    image_answers =[]
    image_predictions =[]
    for id in ids_to_test:
        random_sample = df_to_test.loc[id]
        prediction =inference_simple(config, random_sample['question'], random_sample['answer'], random_sample['image_path'], device,model)
        sep = os.sep
        image_name = random_sample['image_path'].split(sep)[-1]
        image_names.append(image_name)
        image_questions.append(random_sample['question'])
        image_answers.append(random_sample['answer'])
        image_predictions.append(prediction)

    prediction_dict = pd.DataFrame({'image_path':image_names , 'Question':image_questions,
                                    'Target':image_answers,'Prediction':image_predictions})
    print(prediction_dict.to_string())
    #run_validation(model, test_dataloader, tokenizer, config, device)





