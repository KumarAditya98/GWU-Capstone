import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import pickle
import torch.nn as nn
import torch.nn.functional as F

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy().head(1)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy().head(1)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

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
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

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


class GELUActivation(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# class BlipAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(BlipAttention, self).__init__()
#         self.dropout = nn.Dropout(p=0.0)
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.projection = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         B, N, C = x.size()
#         qkv = self.qkv(x).view(B, N, 3, C)  # Reshape without permute
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Change indexing for q, k, v
#         attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
#         attn_weights = F.softmax(attn_weights, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#         x = torch.matmul(attn_weights, v)
#         x = x.transpose(1, 2).reshape(B, N, C)  # Transpose and reshape together
#         x = self.projection(x)
#         return x
#
class BlipAttention(nn.Module):
    def __init__(self, embed_dim):
        super(BlipAttention, self).__init__()
        self.dropout = nn.Dropout(p=0.0)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, encoder_output=None):
        B, N, C = x.size()
        qkv = self.qkv(x).view(B, N, 3, C)  # Reshape without permute
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Change indexing for q, k, v
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2).reshape(B, N, C)  # Transpose and reshape together
        x = self.projection(x)
        return x

class BlipMLP(nn.Module):
    def __init__(self, embed_dim):
        super(BlipMLP, self).__init__()
        self.activation_fn = GELUActivation()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

class BlipEncoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super(BlipEncoderLayer, self).__init__()
        self.self_attn = BlipAttention(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mlp = BlipMLP(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_res = x
        x = self.self_attn(x)
        x = x_res + x
        x = self.layer_norm1(x)
        x_res = x
        x = self.mlp(x)
        x = x_res + x
        x = self.layer_norm2(x)
        return x



class BlipDecoderLayer(nn.Module):
    def __init__(self, embed_dim):
        super(BlipDecoderLayer, self).__init__()
        self.self_attn = BlipAttention(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = BlipAttention(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = BlipMLP(embed_dim)

    def forward(self, x, encoder_output):
        # Self-attention
        x_res = x
        x = self.self_attn(x)
        x = x_res + x
        x = self.layer_norm1(x)

        # Cross-attention
        x_res = x
        x = self.cross_attn(x, encoder_output)
        x = x_res + x
        x = self.layer_norm2(x)

        # Position-wise feedforward network
        x_res = x
        x = self.mlp(x)
        x = x_res + x

        return x


class BlipVisionModel(nn.Module):
    def __init__(self, embed_dim=768, num_layers=12):
        super(BlipVisionModel, self).__init__()
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=(16, 16), stride=(16, 16))
        self.encoder = nn.ModuleList([BlipEncoderLayer(embed_dim) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([BlipDecoderLayer(embed_dim) for _ in range(num_layers)])  # Add decoder layers
        self.post_layernorm = nn.LayerNorm(embed_dim)
        self.reconstruct_conv = nn.ConvTranspose2d(embed_dim, 3, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        encoder_outputs = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x.clone())

        # Decoder
        decoder_output = x
        for layer, encoder_output in zip(self.decoder, reversed(encoder_outputs)):
            decoder_output = layer(decoder_output, encoder_output)

        decoder_output = self.post_layernorm(decoder_output)
        reconstructed_image = self.reconstruct_conv(decoder_output.view(B, -1, H, W))
        return reconstructed_image
    # def forward(self, x):
    #     x = self.patch_embedding(x)
    #     B, C, H, W = x.shape
    #     x = x.flatten(2).transpose(1, 2)
    #
    #     encoder_outputs = []  # Store encoder outputs for cross-attention in decoder
    #
    #     # Encoder
    #     for layer in self.encoder:
    #         x = layer(x)
    #         encoder_outputs.append(x.clone())  # Store encoder output
    #
    #     # Decoder
    #     for layer, encoder_output in zip(self.decoder, reversed(encoder_outputs)):  # Pass encoder output to decoder
    #         x = layer(x, encoder_output)
    #
    #     x = x.transpose(1, 2)#.reshape(B, H // 16, W // 16, C)
    #     #x = x.permute(0, 3, 1, 2)
    #     x = self.post_layernorm(x)
    #     x = x.view(B, C, H, W)
    #     return x


#%%
def model_definition():
    #model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")
    # model.to(device)
    model = BlipVisionModel()
    model = model.to(device)
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
    model.train()


    for epoch in range(num_epochs):
        # --Start Model Training--
        epoch_loss = 0
        train_loss = 0
        steps_train = 0
        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            for step, batch in enumerate(train_gen):
                pixel_values = batch['pixel_values'].to(device)
                reconstructed_images = model(pixel_values)
                print("done")
                loss = compute_loss(reconstructed_images, pixel_values, loss_type='mse')
                epoch_loss += loss.item()
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)
                steps_train += 1
                avg_train_loss = epoch_loss / steps_train
                pbar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

        model.eval()
        eval_loss = 0
        steps_test = 0
        with tqdm(total=len(val_gen), desc=f'Epoch {epoch}') as pbar:
            with torch.no_grad():
                for step, batch in enumerate(train_gen):
                    pixel_values = batch.pop('pixel_values').to(device)
                    reconstructed_images = model(pixel_values)
                    loss = compute_loss(reconstructed_images, pixel_values, loss_type='mse')
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
            #model.save_pretrained("Model/encoder-decoder-blip", from_pt=True)
            torch.save(model.state_dict(), 'Model/encoder-decoder-blip.pth')
            print("Saved model to Model/encoder-decoder-blip")
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
