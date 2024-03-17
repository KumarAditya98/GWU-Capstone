import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import pickle

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy().head(10)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy().head(10)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

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

        #get labels

        if self.type_data == 'train':
            question = xdf_dset.question.get(ID)
            answer = xdf_dset.answer.get(ID)
            image_path = xdf_dset.image_path.get(ID)
            #y = [xdf_dset.target.get(ID)]
            #file = xdf_dset.destination_path.get(ID)
        elif self.type_data == 'test':
            question = xdf_dset_test.question.get(ID)
            answer = xdf_dset_test.answer.get(ID)
            image_path = xdf_dset_test.image_path.get(ID)
        # elif self.type_data == 'dev':
        #     pass
            # y = [xdf_dset_dev.target.get(ID)]
            # file = xdf_dset_dev.destination_path.get(ID)
        #y= torch.FloatTensor(y)
        image = Image.open(image_path).convert('RGB')
        #X = self.processor(images=img, return_tensors='pt')['pixel_values'].squeeze()

        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding
        #return X, question,answer
class CustomDataLoader:
    def __init__(self,config):
        self.BATCH_SIZE = config['BATCH_SIZE']

    def read_data(self):
        list_of_ids = list(xdf_dset.index)
        list_of_ids_test = list(xdf_dset_test.index)
        #list_of_ids_dev = list(xdf_dset_dev.index)

        partition = {
            'train': list_of_ids,
            'test': list_of_ids_test
            # 'dev' : list_of_ids_dev
        }

        params = {'batch_size': self.BATCH_SIZE, 'shuffle': True}

        training_set = CustomDataset(partition['train'], 'train')
        training_generator = data.DataLoader(training_set, **params)

        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params)

        # params = {'batch_size': BATCH_SIZE, 'shuffle': False}
        # dev_set = CustomDataset(partition['dev'], 'dev')
        # dev_generator = data.DataLoader(dev_set, **params)

        return training_generator, test_generator#, dev_generator

if __name__ == '__main__':
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)

    data_loader = CustomDataLoader(config)
    train_gen, val_gen = data_loader.read_data()

    """
    Reference :  https://github.com/dino-chiio/blip-vqa-finetune/tree/main
    """
    #model = model.half()
    model = model.to(device)
    num_epochs = config["EPOCH"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    scaler = torch.cuda.amp.GradScaler()
    tracking_information = []
    patience = 10
    min_eval_loss = float("inf")
    early_stopping_hook = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        #model.train()
        for idx, batch in zip(tqdm(range(len(train_gen)), desc='Training batch: ...'), train_gen):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_masked = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_masked,
                            labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        eval_loss = 0
        for idx, batch in zip(tqdm(range(len(val_gen)), desc='Validating batch: ...'), val_gen):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_masked = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_masked,
                            labels=labels)

            loss = outputs.loss
            eval_loss += loss.item()

        tracking_information.append(
            (epoch_loss / len(train_gen), eval_loss / len(val_gen), optimizer.param_groups[0]["lr"]))
        print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch + 1, epoch_loss / len(train_gen),
                                                                              eval_loss / len(val_gen),
                                                                              optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if eval_loss < min_eval_loss:
            model.save_pretrained("Model/blip-saved-model", from_pt=True)
            print("Saved model to Model/blip-saved-model")
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break
    pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
    print("The finetuning process has done!")