import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
#MODEL_FOLDER = CODE_DIR + os.sep + 'main_code' +os.sep +'Model'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()#.head(100)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy()#.head(100)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
patch_embedding_weight_init = blip_model.state_dict()['vision_model.embeddings.patch_embedding.weight']
patch_embedding_bias_init = blip_model.state_dict()['vision_model.embeddings.patch_embedding.bias']


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
        self.transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        #self.processor = processor
        #self.processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        ID = self.list_IDs[index]
        if self.type_data == 'train':
            # question = xdf_dset.question.get(ID)
            # answer = xdf_dset.answer.get(ID)
            image_path = xdf_dset.image_path.get(ID)

        elif self.type_data == 'test':
            # question = xdf_dset_test.question.get(ID)
            # answer = xdf_dset_test.answer.get(ID)
            image_path = xdf_dset_test.image_path.get(ID)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

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

def init_conv_weights(m):
    if isinstance(m, nn.Conv2d):

        #in_channels, out_channels, kernel_size, kernel_size = m.weight.shape
        #reshaped_weights = patch_embedding_weight_init.reshape(out_channels, in_channels, kernel_size, kernel_size)
        m.weight.data.copy_(patch_embedding_weight_init)

        #reshaped_bias = patch_embedding_bias_init.reshape(-1)
        m.bias.data.copy_(patch_embedding_bias_init)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 768, kernel_size=(16,16), stride=(16,16))
        self.conv2 = nn.Conv2d(768, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Initializing conv1 with BLIP weights
        self.conv1.apply(init_conv_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 768, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(768, 3, kernel_size=16, stride=16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
#%%
def model_definition():
    model = ConvAutoencoder()
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
                pixel_values = batch.to(device)
                reconstructed_images = model(pixel_values)
                #print("done")
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
                for step, batch in enumerate(val_gen):
                    pixel_values = batch.to(device)
                    reconstructed_images = model(pixel_values)
                    loss = compute_loss(reconstructed_images, pixel_values, loss_type='mse')
                    eval_loss += loss.item()
                    steps_test+=1
                    pbar.update(1)
                    avg_test_loss = eval_loss / steps_test
                    pbar.set_postfix_str(f'Validation  Loss: {avg_test_loss:.5f}')

        tracking_information.append(
            (epoch_loss / len(train_gen), eval_loss / len(val_gen), optimizer.param_groups[0]["lr"]))
        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_gen),
                                                                              eval_loss / len(val_gen),
                                                                              optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if eval_loss < min_eval_loss:
            #model_file = MODEL_FOLDER + os.sep + 'blip-conv-patch-embedding-finetune.pth'
            torch.save(model.state_dict(), 'Model/blip-conv-patch-embedding-finetune.pth')
            print("Saved model to Model/blip-conv-patch-embedding-finetune")
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
