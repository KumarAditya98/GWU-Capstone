import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer#, TransformerEncoderLayer, TransformerDecoderLayer
from transformers import BertModel
''''
Reference : https://github.com/tezansahu/VQA-With-Multimodal-Transformers/blob/main/src/model.py#L7
'''
# print(tf.config.list_physical_devices("GPU"))
CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()#.head(100)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy()#.head(100)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
text_encoder_model = BertModel.from_pretrained('bert-base-uncased')
from transformers import AutoModel, BlipForQuestionAnswering#, Transformer, TransformerDecoderLayer
#decoder = blip_model.text_decoder
#question_encoder = BertEncoder()
#configuration = BlipConfig()

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class CustomDataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.processor = processor
        #self.processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.type_data == 'train':
            question = xdf_dset.question.get(ID)
            answer = xdf_dset.answer.get(ID)
            image_path = xdf_dset.image_path.get(ID)

        elif self.type_data == 'test':
            question = xdf_dset_test.question.get(ID)
            answer = xdf_dset_test.answer.get(ID)
            image_path = xdf_dset_test.image_path.get(ID)

        image = Image.open(image_path).convert('RGB')

        #image embedding
        image_embedding = self.processor(images=image, return_tensors="pt")['pixel_values']
        question_encoding = self.processor(text=question,padding="max_length", return_tensors="pt")
        # question_encoder = BertEncoder(self.processor.text_model)
        # question_embedding = question_encoder(input_ids, attention_mask)
        #print(f"answer : {answer}")
        answer = str(answer)
        answer_encoding = self.processor.tokenizer.encode(
            answer, max_length=20, pad_to_max_length=True, return_tensors='pt'
        )

        dictionary = {'image_embedding':image_embedding,'question_encoding':question_encoding,'answer_encoding':answer_encoding}
        return dictionary
        #return image_embedding, question_encoding, answer_encoding

class CustomDataLoader:
    def __init__(self,config):
        self.BATCH_SIZE = config['BATCH_SIZE']

    def read_data(self):
        list_of_ids = list(xdf_dset.index)
        list_of_ids_test = list(xdf_dset_test.index)
        partition = {
            'train': list_of_ids,
            'test': list_of_ids_test
        }
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': True}
        training_set = CustomDataset(partition['train'], 'train')
        training_generator = data.DataLoader(training_set, **params)
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params)
        return training_generator, test_generator#, dev_generator


#         # Decoder
#         # output = self.decoder(tgt=target_ids, memory=fused_output)[0]
#         # logits = self.output_layer(output)
#         #
#         # output = {"logits": logits}
#         #
#         # if target_ids is not None:
#         #     loss = self.criterion(logits.view(-1, self.num_labels), target_ids.view(-1))
#         #     output["loss"] = loss
#
#         return output

class MultimodalVQAModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        text_encoder_name: str,
        #image_encoder_name: str,
        decoder_hidden_size: int,
        decoder_num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.text_encoder = text_encoder_model#BertModel.from_pretrained('bert-base-uncased')#AutoModel.from_pretrained(text_encoder_name)
        self.image_encoder = blip_model.vision_model

        # self.fusion = nn.Sequential(
        #     nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, decoder_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        # )
        # self.fusion_cls = nn.Sequential(
        #     nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, decoder_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        # )
        self.text_decoder = blip_model.text_decoder
        # self.decoder = Transformer(
        #     d_model=decoder_hidden_size,
        #     nhead=8,
        #     num_encoder_layers=0,
        #     num_decoder_layers=decoder_num_layers,
        #     dim_feedforward=decoder_hidden_size * 4,
        #     dropout=dropout,
        #     activation="relu",
        # )
        #
        # self.output_layer = nn.Linear(decoder_hidden_size, self.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        target_ids: torch.Tensor = None,
    ):
        ip = input_ids.squeeze(1)
        at =attention_mask.squeeze(1)
        pv =pixel_values.squeeze(1)
        tg = target_ids.squeeze(1)
        encoded_text = self.text_encoder(input_ids=ip, attention_mask=at, return_dict=True)
        encoded_image = self.image_encoder(pixel_values=pv, return_dict=True)

        length_difference = encoded_image["last_hidden_state"].shape[1] - encoded_text["last_hidden_state"].shape[1]

        # Pad the text encoding with zeros to match the length of the image encoding
        if length_difference > 0:
            #pad_left = length_difference // 2
            pad_right = length_difference# - pad_left
            encoded_text["last_hidden_state"] = F.pad(encoded_text["last_hidden_state"], (0, 0, 0, pad_right))
        else:
            encoded_text["last_hidden_state"] = encoded_text["last_hidden_state"][:,
                                                :encoded_image["last_hidden_state"].shape[1], :]

        # fused_output_cls = self.fusion(
        #     torch.cat([encoded_text["pooler_output"], encoded_image["pooler_output"]], dim=1)
        # )
        #fused_output = torch.cat([encoded_text["last_hidden_state"], encoded_image["last_hidden_state"]], dim=-1)
        fused_output = encoded_text["last_hidden_state"] + encoded_image["last_hidden_state"]
        decoder_output = self.text_decoder(
            input_ids=tg[:, :-1],
            encoder_hidden_states=fused_output,
            encoder_attention_mask=None,
            labels=tg[:, 1:],
            return_dict=True
        )
        #SGD try -> multiple epoch
        ouput = decoder_output
        # output = decoder_output[1]
        # logits = decoder_output['logits']
        # predicted_token_ids = torch.argmax(logits, dim=-1)

        #combined_embedding = torch.cat([encoded_text["pooler_output"], encoded_image["pooler_output"]], dim=1)

        # Decoder
        # output = self.decoder(fused_output)
        # output = self.output_layer(output)
        # Prepare the decoder input

        # Prepare the decoder input
        #answer_embedding = self.text_encoder.embeddings(tg)

        # Pass the fused output and answer embedding through the decoder
        # decoder_output = self.decoder(
        #     answer_embedding,
        #     fused_output,
        #     fused_output,
        #     memory_key_padding_mask=~(tg > 0)
        # )
        # output = decoder_output[1]
        # output = self.output_layer(decoder_output)
        # decoder_input = self.text_encoder.embeddings(target_ids)
        # decoder_output = self.decoder(
        #     decoder_input,
        #     fused_output,
        #     memory_key_padding_mask=~(target_ids > 0)
        # )
        return decoder_output
def model_definition():

    num_labels = 30524
    text_encoder_name = "bert-base-uncased"
    image_encoder_name = "resnet50"
    decoder_hidden_size = 768
    decoder_num_layers = 2
    dropout = 0.1

    model = MultimodalVQAModel(
    num_labels=num_labels,
    text_encoder_name=text_encoder_name,
    #image_encoder_name=image_encoder_name,
    decoder_hidden_size=decoder_hidden_size,
    decoder_num_layers=decoder_num_layers,
    dropout=dropout,
    )
    #model = MultimodalVQAModel()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scheduler, scaler
def compute_loss(reconstructed_images, original_images, loss_type='mse'):
    if loss_type == 'mse':
        loss = nn.MSELoss()(reconstructed_images, original_images)
    elif loss_type == 'mae':
        loss = nn.L1Loss()(reconstructed_images, original_images)
    else:
        raise ValueError("Unsupported loss type. Please choose 'mse' or 'mae'.")
    return loss

def train_test(train_gen, val_gen ,config):
    patience = config['patience']
    num_epochs = config["EPOCH"]
    min_eval_loss = float("inf")
    tracking_information = []
    model, optimizer, scheduler, scaler = model_definition()



    for epoch in range(num_epochs):
        # --Start Model Training--
        epoch_loss = 0
        train_loss = 0
        steps_train = 0
        model.train()
        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            for step, batch in enumerate(train_gen):
                #image_embedding, question_encoding, answer_encoding
                image_embedding = batch['image_embedding'].to(device)
                question_encoding = batch.pop('question_encoding').to(device)
                answer_encoding = batch.pop('answer_encoding').to(device)
                output = model(
                    input_ids=question_encoding['input_ids'],
                    pixel_values=image_embedding,
                    attention_mask=question_encoding['attention_mask'],
                    target_ids=answer_encoding
                )
                #reconstructed_images = model(image_embedding,question_encoding,answer_encoding)
                #print("done")
                #loss = compute_loss(reconstructed_images, pixel_values, loss_type='mse')
                #output = model(pixel_values)
                loss = output.loss
                epoch_loss += loss.item()
                optimizer.zero_grad()

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                pbar.update(1)
                steps_train += 1
                avg_train_loss = epoch_loss / steps_train
                pbar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

        model.eval()
        eval_loss = 0
        steps_test = 0
        with tqdm(total=len(val_gen), desc=f'Epoch {epoch}') as pbar:
            with torch.no_grad():
                for step, batch in enumerate(val_gen):
                    # pixel_values = batch.pop('pixel_values').to(device)
                    # input_ids = batch.pop('input_ids').to(device)
                    # output = model(pixel_values)
                    image_embedding = batch['image_embedding'].to(device)
                    question_encoding = batch.pop('question_encoding').to(device)
                    answer_encoding = batch.pop('answer_encoding').to(device)
                    output = model(
                        input_ids=question_encoding['input_ids'],
                        pixel_values=image_embedding,
                        attention_mask=question_encoding['attention_mask'],
                        target_ids=answer_encoding
                    )
                    loss = output.loss
                    #loss = compute_loss(reconstructed_images, pixel_values, loss_type='mse')
                    eval_loss += loss.item()
                    steps_test+=1
                    pbar.update(1)
                    avg_test_loss = eval_loss / steps_test
                    pbar.set_postfix_str(f'Test  Loss: {avg_test_loss:.5f}')

        tracking_information.append(
            (epoch_loss / len(train_gen), eval_loss / len(val_gen), optimizer.param_groups[0]["lr"]))
        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_gen),
                                                                              eval_loss / len(val_gen),
                                                                              optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if eval_loss < min_eval_loss:
            #torch.save(model.state_dict(), 'Model/blip-encoder-blip.pth')
            print("Saved model to Model/encoder-decoder-transformer")
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break

if __name__ == '__main__':
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)

    data_loader = CustomDataLoader(config)
    train_gen, val_gen = data_loader.read_data()

    train_test(train_gen, val_gen,  config)
    """
    Reference :  https://github.com/dino-chiio/blip-vqa-finetune/tree/main
    """
