import numpy as np
from custom_image_question_answer import build_transformer
from dataset import QuestionAnswerDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
#from transformers import  BlipForQuestionAnswering
import os
import pandas as pd
import logging
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


CUR_DIR = os.getcwd() #custom_archi
MAIN_CODE_DIR = os.path.dirname(CUR_DIR) #main_code
CODE_DIR = os.path.dirname(MAIN_CODE_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR) #main_code
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
TOKENIZER_File = CONFIG_FOLDER + os.sep + 'custom_tokenizer.json'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
train_ds_raw = xdf_data[xdf_data["split"] == 'train'].copy().reset_index()#.head(20)
val_ds_raw = xdf_data[xdf_data["split"] == 'val'].copy().reset_index()#.head(20)
#blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

#from transformers import  BlipForQuestionAnswering
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer

import torchmetrics
#from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

def get_dataset_tokenizer(config):
    if os.path.exists(TOKENIZER_File):
        tokenizer = Tokenizer.from_file(str(TOKENIZER_File))
    else:
        print(f"Tokenizer file not found at {TOKENIZER_File}. Creating a new tokenizer...")
        train_questions = list(train_ds_raw['question'].values)
        train_answers = list(train_ds_raw['answer'].values)
        train_answers = [str(x) for x in train_answers]
        all_data = train_questions.copy()
        all_data.extend(train_answers)
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(min_frequency=2,
                             special_tokens=["[SOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train_from_iterator(all_data, trainer)
        tokenizer.save(TOKENIZER_File)
        print(f"Tokenizer created and saved at {TOKENIZER_File}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_File))

    train_ds = QuestionAnswerDataset(train_ds_raw, tokenizer,config['answer_seq_len'],config['question_seq_len'])
    val_ds = QuestionAnswerDataset(val_ds_raw, tokenizer,config['answer_seq_len'],config['question_seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def model_definition(config, vocab_len):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_transformer(vocab_len, config["question_seq_len"], config['answer_seq_len'], d_model=config['d_model'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    preload = config['preload']
    initial_epoch =0
    global_step =0
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
                                                                                                        preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    return model,optimizer,initial_epoch, global_step

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    device = torch.device(device)
    train_dataloader, val_dataloader, tokenizer = get_dataset_tokenizer(config)
    model, optimizer, intital_epoch, global_step = model_definition(config, tokenizer.get_vocab_size())

    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    batch_size = config['batch_size']
    #initial_epoch = 0
    #global_step = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    #criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #loss = criterion(model_out.view(-1, model_out.size(-1)), answer_input["input_ids"].view(-1))
    for epoch in range(intital_epoch, config['num_epochs']):
        global_step+=1
        torch.cuda.empty_cache()
        model.train()
        step_train =0
        sum_epoch_loss= 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            question_input = batch['question_input'].to(device)  # (b, seq_len)
            answer_input = batch['answer_input'].to(device)  # (B, seq_len)
            question_mask = batch['question_mask'].to(device)  # (B, 1, 1, seq_len)
            answer_mask = batch['answer_mask'].to(device)  # (B, 1, seq_len, seq_len)
            pixel_values = batch['pixel_values'].to(device)

            encoder_output,image_embed = model.encode(question_input, question_mask,pixel_values=pixel_values.squeeze(1))  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, question_mask, answer_input,
                                          answer_mask,image_embed)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)
            # each row represents a single prediction and each column represents a class (vocabulary token)(b*seq_len, vocab_size)
            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            sum_epoch_loss = sum_epoch_loss+ loss.item()
            step_train+=1
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            #global_step += 1.log
        avg_train_loss = sum_epoch_loss/step_train
        logging.info("==========train loss========")
        logging.info(f"Train Loss: {avg_train_loss:.4f} at epoch {epoch}")
        logging.info("==========train loss========")

        model.eval()
        question_texts = []
        expected = []
        predicted = []
        val_epoch_loss = 0.0
        val_epoch_step_size = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                question_input = batch["question_input"].to(device)  # (b, seq_len)
                question_mask = batch["question_mask"].to(device)  # (b, 1, 1, seq_len)
                answer_input = batch['answer_input'].to(device)
                answer_mask = batch['answer_mask'].to(device)
                pixel_values = batch["pixel_values"].to(device)
                question_text = batch["question_text"]#.to(device)
                answer_text = batch["answer_text"]#.to(device)

                encoder_output, image_embed = model.encode(question_input, question_mask,
                                                           pixel_values=pixel_values.squeeze(1))
                ###################### decoder #########################################

                decoder_input = torch.empty(batch_size, 1).fill_(sos_idx).type_as(question_input).to(device) #(b,1) sos
                decoder_outputs = torch.empty(batch_size, 0).type_as(question_input).long().to(device) #(batch_size, 0)

                while True:
                    if decoder_input.size(1) == config['answer_seq_len']:
                        break

                    # build mask for target
                    decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)
                    decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, 1, -1, -1)
                    out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask, image_embed) #next token prob

                    # get next token
                    prob = model.project(out[:, -1])
                    _, next_word = torch.max(prob, dim=1)
                    decoder_outputs = torch.cat([decoder_outputs, next_word.unsqueeze(-1)], dim=1)
                    decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=1)
                    if (next_word == eos_idx).all() or decoder_input.size(1) == config['answer_seq_len']:#all have eos or max seq len reached
                        break
                model_out = decoder_outputs

                #question_text_batch = batch["question_text"]
                #answer_text_batch = batch["answer_text"]
                model_out_text_batch = [tokenizer.decode(output.detach().cpu().numpy()) for output in model_out]
                question_texts.extend(question_text)
                expected.extend(answer_text)
                predicted.extend(model_out_text_batch)

                ###################decoder end
                # Generate the whole answer sequence in one shot
                # decoder_output = model.decode(encoder_output, question_mask, answer_input, answer_mask, image_embed)
                # proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)
                #
                # # Compare the output with the label
                # label = batch["label"].to(device)  # (B, seq_len)
                # loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                #
                #val_epoch_loss += loss.item()
                #val_epoch_step_size +=1
                # # Convert model output to text
                # model_out = torch.argmax(proj_output, dim=-1)  # (B, seq_len)

                # model_out_text_batch = [tokenizer.decode(output.detach().cpu().numpy()) for output in model_out]
                #
                # question_texts.extend(question_text)
                # expected.extend(answer_text)
                # predicted.extend(model_out_text_batch)
        #avg_val_loss = val_epoch_loss / val_epoch_step_size

        #logging.info("==========val loss========")
        #logging.info(f"Val Loss: {avg_val_loss:.4f} at epoch {epoch}")
        #logging.info("==========val loss========")

        ############ Evaluation metric ######
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        logging.info(f"Validation CER: {cer:.4f} ")
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        logging.info(f"Validation WER: {wer:.4f} ")

        list_of_metrics = ['bleu', 'rouge', 'jac', 'em', 'f1', 'meteor']
        res_dict = metrics_func(list_of_metrics, y_true=expected, y_pred=predicted)
        res_str = ""
        for key, value in res_dict.items():
            res_str = res_str +"\n"+ key +" "+ str(round(value,2))
            logging.info(f"Validation {key}: {value:.4f} ")
        print(res_str)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        ############ Evaluation metric ######
                ###################### decoder #########################################
                # encoder_output, image_embed = model.encode(question_input, question_mask,
                #                                            pixel_values=pixel_values.squeeze(1))
                # decoder_input = torch.empty(batch_size, 1).fill_(sos_idx).type_as(question_input).to(device) #(b,1) sos
                # decoder_outputs = torch.empty(batch_size, 0).type_as(question_input).long().to(device) #(batch_size, 0)
                #
                # while True:
                #     if decoder_input.size(1) == config['answer_seq_len']:
                #         break
                #
                #     # build mask for target
                #     decoder_mask = causal_mask(decoder_input.size(1)).type_as(question_mask).to(device)
                #     decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, 1, -1, -1)
                #     out = model.decode(encoder_output, question_mask, decoder_input, decoder_mask, image_embed) #next token prob
                #
                #     # get next token
                #     prob = model.project(out[:, -1])
                #     _, next_word = torch.max(prob, dim=1)
                #     decoder_outputs = torch.cat([decoder_outputs, next_word.unsqueeze(-1)], dim=1)
                #     decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=1)
                #     if (next_word == eos_idx).all() or decoder_input.size(1) == config['answer_seq_len']:#all have eos or max seq len reached
                #         break
                # model_out = decoder_outputs
                #
                # question_text_batch = batch["question_text"]
                # answer_text_batch = batch["answer_text"]
                # model_out_text_batch = [tokenizer.decode(output.detach().cpu().numpy()) for output in model_out]
                # #
                # question_texts.extend(question_text_batch)
                # expected.extend(answer_text_batch)
                # predicted.extend(model_out_text_batch)


        # model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'global_step': global_step
        # }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
