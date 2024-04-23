from custom_image_question_answer import build_transformer

from config import get_config, get_weights_file_path, latest_weights_file_path
#from transformers import  BlipForQuestionAnswering
import os
import pandas as pd
import torch
import logging
from dataset import QuestionAnswerDataset, causal_mask
from tokenizers import Tokenizer
from PIL import Image
from transformers import  BlipForQuestionAnswering,BlipProcessor
blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")


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
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
#train_ds_raw = xdf_data[xdf_data["split"] == 'train'].copy().reset_index()#.head(100)
val_ds_raw = xdf_data[xdf_data["split"] == 'val'].copy().reset_index()


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
    return model
def inference_simple(config,question_text,answer_text,image_path,device):

    tokenizer = Tokenizer.from_file(str(TOKENIZER_File))
    image = Image.open(image_path).convert('RGB')
    image_embedding = blipprocessor(image, return_tensors="pt")
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    question_tokens = tokenizer.encode(question_text).ids
    answer_tokens = tokenizer.encode(answer_text).ids
#train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # image = Image.open(image_path).convert('RGB')
    # image_embed = blipprocessor(image, return_tensors="pt")

    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
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

    #encoder_output,image_embed = model.encode(question_input, question_mask,pixel_values=image_embedding.squeeze(1))


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
