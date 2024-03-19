import os
from ruamel.yaml import YAML
import pandas as pd
from torch.utils import data
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
import pickle
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoModelForCausalLM
from transformers import Blip2Processor




CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train']
xdf_dset_test = xdf_data[xdf_data["split"] == 'val']

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")

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
        # encoding = self.processor(image, question, padding=True, truncation=True, return_tensors="pt")
        encoding = self.processor(question, image, padding=True, truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=encoding.input_ids.shape[1], pad_to_max_length=True, return_tensors='pt'
        )

        encoding["labels"] = labels
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding

# rom torch.utils.data import DataLoader
#
def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  labels = [item['labels'] for item in batch]
  input_ids_lists = [ids.tolist() for ids in input_ids]
  pixel_values_lists = [px.tolist() for px in pixel_values]
  max_length = max(len(ids) for ids in input_ids_lists)

  # Pad input_ids and attention_mask to the same length
  padded_input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids_lists]
  padded_attention_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in attention_mask]

  # Convert padded_input_ids and padded_attention_mask back to tensors
  padded_input_ids = torch.tensor(padded_input_ids)
  padded_attention_mask = torch.tensor(padded_attention_mask)

  # Stack the padded tensors
  batch = {}
  batch['input_ids'] = padded_input_ids
  batch['attention_mask'] = padded_attention_mask
  batch['pixel_values'] = torch.tensor(pixel_values_lists)
  batch['labels'] = torch.stack(labels)

  return batch

# def collate_fn(batch):
#     # Extract input_ids and attention_mask tensors from batch
#     input_ids = [item['input_ids'] for item in batch]
#     attention_mask = [item['attention_mask'] for item in batch]
#
#     # Determine the maximum length of input_ids
#     max_length = max(ids.size(1) for ids in input_ids)
#
#     # Pad input_ids and attention_mask to the same length
#     padded_input_ids = torch.stack([torch.cat([ids, torch.zeros((1, max_length - ids.size(1)), dtype=torch.long)]) for ids in input_ids])
#     padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros((1, max_length - mask.size(1)), dtype=torch.long)]) for mask in attention_mask])
#
#     # Extract pixel_values and labels tensors from batch
#     pixel_values = torch.stack([item['pixel_values'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])
#
#     # Return the processed batch
#     return {
#         'input_ids': padded_input_ids,
#         'attention_mask': padded_attention_mask,
#         'pixel_values': pixel_values,
#         'labels': labels
#     }

# def collate_fn(batch):
#     # pad the input_ids and attention_mask
#     processed_batch = {}
#     for key in batch[0].keys():
#         if key in ["pixel_values",'input_ids']:
#             processed_batch[key] = torch.stack([example[key] for example in batch])
#         #     processed_batch["input_ids"] = text_inputs["input_ids"]
#         #     processed_batch["attention_mask"] = text_inputs["attention_mask"]
#         # elif key == 'question':
#         #     text_inputs = processor.tokenizer(
#         #         [example["question"] for example in batch], padding=True, return_tensors="pt"
#         #     )
#         #
#         # elif key == 'labels':
#         #     # No need to stack labels here, already stacked during encoding
#         #     processed_batch[key] = torch.stack([example[key] for example in batch])
#     # processed_batch["input_ids"] = text_inputs["input_ids"]
#     # processed_batch["attention_mask"] = text_inputs["attention_mask"]
#     return processed_batch


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
        training_generator = data.DataLoader(training_set, **params, collate_fn = collate_fn)
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params)
        return training_generator, test_generator#, dev_generator


def model_definition(config):
    #model = model
    # model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = AutoModelForVisualQuestionAnswering.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    # )
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scheduler, scaler

def train_test(train_gen, val_gen ,config):
    patience = config['patience']
    num_epochs = config["EPOCH"]
    min_eval_loss = float("inf")
    tracking_information = []
    model, optimizer, scheduler, scaler = model_definition(config)

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
            model.save_pretrained("Model/git-saved-model", from_pt=True)
            print("Saved model to Model/git-saved-model")
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
