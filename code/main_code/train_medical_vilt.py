import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import pickle
import cv2

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()#.head(200)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy()#.head(100)

#processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

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
    def __init__(self, list_IDs, type_data, processor):
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

        # image = Image.open(image_path).convert('RGB')
        # #image = image.resize((250, 250), Image.ANTIALIAS)
        # #image = cv2.imread(image_path)
        # #image = cv2.resize(image,(384,384))
        # #encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        # encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        # # labels = self.processor.tokenizer.encode(
        # #     answer, max_length = 3129, pad_to_max_length=True, return_tensors='pt'
        # # )
        #image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        targets = torch.zeros(len(config.id2label))
        label = config.label2id[answer]
        #targets = torch.zeros(len(config.id2label))
        # for label, score in zip(labels, scores):
        #       targets[label] = score
        # encoding["labels"] = targets
        #for label, score in zip(labels, scores):
        targets[label] = 1
        encoding["labels"] = targets
        # for k,v in encoding.items():
        #     encoding[k] = v.squeeze()
        return encoding


def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch
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
        training_set = CustomDataset(partition['train'], 'train',processor)
        training_generator = data.DataLoader(training_set, **params,collate_fn=collate_fn)
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test',processor)
        test_generator = data.DataLoader(test_set, **params,collate_fn=collate_fn)
        return training_generator, test_generator#, dev_generator


def model_definition(config_yml):
    #model = model
#    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                     id2label=config.id2label,
                                                     label2id=config.label2id)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scheduler, scaler



def train_test(train_gen, val_gen ,config_yml):
    patience = config_yml['patience']
    num_epochs = config_yml["EPOCH"]
    min_eval_loss = float("inf")
    tracking_information = []
    model, optimizer, scheduler, scaler = model_definition(config_yml)

    for epoch in range(num_epochs):
        # --Start Model Training--
        epoch_loss = 0
        train_loss = 0
        steps_train = 0
        model.train()
        with tqdm(total=len(train_gen), desc=f'Epoch {epoch}') as pbar:
            for step, batch in enumerate(train_gen):
                input_ids = batch.pop('input_ids').to(device)
                pixel_values = batch.pop('pixel_values').to(device)
                attention_masked = batch.pop('attention_mask').to(device)
                labels = batch.pop('labels').to(device)
                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                attention_mask=attention_masked,
                                labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(1)
                steps_train+=1
                avg_train_loss = epoch_loss / steps_train
                pbar.set_postfix_str(f'Train Loss: {avg_train_loss:.5f}')

        model.eval()
        eval_loss = 0
        steps_test = 0
        with tqdm(total=len(val_gen), desc=f'Epoch {epoch}') as pbar:
            with torch.no_grad():
                for step, batch in enumerate(train_gen):
        # for idx, batch in zip(tqdm(range(len(val_gen)), desc='Validating batch: ...'), val_gen):
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
            model.save_pretrained("Model/vilt-saved-model", from_pt=True)
            print("Saved model to Model/vilt-saved-model")
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                break


def set_label_ids():
    ViltConfig_config_filename = CONFIG_FOLDER + os.sep + 'ViltConfig.pkl'
    unique_labels = set(xdf_dset['answer'].values)
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    config.id2label = id2label
    config.label2id = label2id
    with open(ViltConfig_config_filename , 'wb') as f:
        pickle.dump(config, f)

if __name__ == '__main__':
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config_yml = yaml.load(file)
    set_label_ids()
    data_loader = CustomDataLoader(config_yml)
    train_gen, val_gen = data_loader.read_data()

    train_test(train_gen, val_gen,  config_yml)
    """
    Reference :  https://github.com/dino-chiio/blip-vqa-finetune/tree/main
    """
