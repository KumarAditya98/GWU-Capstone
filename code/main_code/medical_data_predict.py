import pandas as pd
import os
from torch.utils import data
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from ruamel.yaml import YAML
import torch
import tqdm

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
test_data_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"
xdf_dset_test = pd.read_excel(test_data_excel_file)

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
        question = xdf_dset_test.question.get(ID)
        answer = xdf_dset_test.answer.get(ID)
        image_path = xdf_dset_test.image_path.get(ID)
        image = Image.open(image_path).convert('RGB')
        encoding = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

        return image_path, encoding, answer

class CustomDataLoader:
    def __init__(self,config,processor,model):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.processor = processor
        self.model = model
    def read_data(self):
        list_of_ids_test = list(xdf_dset_test.index)
        partition = {
            'test': list_of_ids_test
        }
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], 'test')
        test_generator = data.DataLoader(test_set, **params)

        return test_generator
if __name__ == '__main__':
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")

    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    test_loader = CustomDataLoader(config,processor,model)
    data = []
    for idx, batch in zip(tqdm(range(len(test_loader)), desc='Validating batch: ...'), test_loader):
        encoding = batch.pop('encoding').to(device)
        image_paths = batch.pop('image_path').to(device)
        answers = batch.pop('answer').to(device)
        out = model.generate(**encoding)
        generated_texts = processor.decode(out[0], skip_special_tokens=True)

        for image_path, answer, generated_text in zip(image_paths, answers, generated_texts):
            data.append({'Image Path': image_path, 'Answer': answer, 'Prediction': generated_text})
    df = pd.DataFrame(data)
