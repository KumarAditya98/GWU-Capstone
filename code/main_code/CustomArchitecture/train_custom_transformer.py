
from custom_image_question_answer import build_transformer
from dataset import QuestionAnswerDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
#from transformers import  BlipForQuestionAnswering
import os
import pandas as pd
import logging
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
train_ds_raw = xdf_data[xdf_data["split"] == 'train'].copy().reset_index()#.head(100)
val_ds_raw = xdf_data[xdf_data["split"] == 'val'].copy().reset_index()#.head(100)
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
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def greedy_decode(model, question, question_mask,pixel_values, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output,image_embed = model.encode(question, question_mask,pixel_values=pixel_values.squeeze(1))
    # alpha = 0.8  # Adjust the weight as needed
    # combined_output = alpha * encoder_output + (1 - alpha) * image_embed
    # encoder_output = combined_output
    # Initialize the decoder input with the sos token
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
def log_message(msg):
    logging.info(msg)

def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg,  num_examples=10):
    model.eval()
    count = 0

    question_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            question_input = batch["question_input"].to(device)  # (b, seq_len)
            question_mask = batch["question_mask"].to(device)  # (b, 1, 1, seq_len)
            pixel_values = batch["pixel_values"].to(device)
            # check that the batch size is 1
            assert question_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, question_input, question_mask,pixel_values, tokenizer, max_len, device)

            question_text = batch["question_text"][0]
            answer_text = batch["answer_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            question_texts.append(question_text)
            expected.append(answer_text)
            predicted.append(model_out_text)

            # Print the question, answer and model output
            #print_msg('-' * console_width)
            # print_msg(f"{f'QUESTION: ':>12}{question_text}")
            # print_msg(f"{f'ANSWER: ':>12}{answer_text}")
            # print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # if count == num_examples:
            #     #print_msg('-' * console_width)
            #     break

    #if writer:
        # Evaluate the character error rate
        # Compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    logging.info(f"Validation CER: {cer:.4f} ")
    #writer.add_scalar('validation cer', cer, global_step)
    #    writer.flush()

        # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
        #writer.add_scalar('validation wer', wer, global_step)
        #writer.flush()
    logging.info(f"Validation WER: {wer:.4f} ")
        # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    #    writer.add_scalar('validation BLEU', bleu, global_step)
    logging.info(f"Validation BLEU: {bleu:.4f} ")
    #    writer.flush()


def get_ds(config):
    # Load your dataset
    # ds_raw = load_dataset(f"{config['datasource']}", split='train')
    #
    # # Build tokenizer
    # tokenizer = Tokenizer(BPE())
    # tokenizer.pre_tokenizer = Whitespace()
    # trainer = BpeTrainer(min_frequency=2,
    #                      special_tokens=["[SOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # tokenizer.train_from_iterator(ds_raw['question'] + ds_raw['answer'], trainer)
    # tokenizer.save(config['tokenizer_file'])
    #
    # # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    tokenizer = Tokenizer.from_file(str(TOKENIZER_File))#tokenizer.save(TOKENIZER_File)
    train_ds = QuestionAnswerDataset(train_ds_raw, tokenizer,config['answer_seq_len'],config['question_seq_len'])#, config['seq_len'])
    val_ds = QuestionAnswerDataset(val_ds_raw, tokenizer,config['answer_seq_len'],config['question_seq_len'])#, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_len):
    #model = build_transformer(vocab_len, vocab_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    model = build_transformer(vocab_len, config["question_seq_len"], config['answer_seq_len'], d_model=config['d_model'])
    return model


def train_model(config):
    # Define the device
    global blip_vision_model
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

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # Tensorboard
    #writer = SummaryWriter(config['experiment_name'])
    # # Configure logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    #
    # # Log the updates
    # logging.info(f"Experiment Name: {config['experiment_name']}")
    # logging.info(f"Epoch: {epoch}")
    # logging.info(f"Loss: {loss.item():.4f}")
    # logging.info(f"Accuracy: {accuracy:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
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

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    #blip_vision_model = blip_vision_model.to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
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
            # Run the tensors through the encoder, decoder and the projection layer
            #image_embedding =blip_vision_model(pixel_values = pixel_values.squeeze(1))
            encoder_output,image_embed = model.encode(question_input, question_mask,pixel_values=pixel_values.squeeze(1))  # (B, seq_len, d_model)
            # alpha = 0.4  # Adjust the weight as needed
            # combined_output = alpha * encoder_output + (1 - alpha) * image_embed
            # encoder_output=combined_output
            decoder_output = model.decode(encoder_output, question_mask, answer_input,
                                          answer_mask,image_embed)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            sum_epoch_loss = sum_epoch_loss+ loss
            step_train+=1
            # Log the loss
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()
            # # logging.info("==========train loss========")
            # logging.info(f"Train Loss: {loss.item():.4f} at global step {global_step}")
            # #logging.info("==========train loss========")
            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            #global_step += 1.log
        avg_train_loss = sum_epoch_loss/step_train
        logging.info("==========train loss========")
        logging.info(f"Train Loss: {avg_train_loss:.4f} at epoch {epoch}")
        logging.info("==========train loss========")
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer, config['answer_seq_len'], device,
                       log_message)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['d_model']  =768
    train_model(config)