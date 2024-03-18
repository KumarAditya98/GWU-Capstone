import pandas as pd
import os
from torch.utils import data
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from ruamel.yaml import YAML
import torch
from tqdm import tqdm
from datetime import datetime

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
test_data_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"

generated_result_folder = EXCEL_FOLDER  + os.sep + 'generated_result'
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not os.path.exists(generated_result_folder):
    os.mkdir(generated_result_folder)
generated_result_excel_file = f"{generated_result_folder}{os.sep}test_data_{current_time}.xlsx"
#generated_result_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"
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
    def __init__(self, list_IDs, processor):
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
        #encoding = self.processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, pad_to_max_length=True, return_tensors='pt'
        )
        #encoding["answers"] = answer
        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding,image_path,question,answer
        #return image_path, encoding, answer
        #return encoding
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
        test_set = CustomDataset(partition['test'], self.processor )
        test_generator = data.DataLoader(test_set, **params)
        return test_generator

if __name__ == '__main__':
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("Model/vilt-saved-model").to("cuda")

    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    test_loader = CustomDataLoader(config,processor,model)
    test_gen = test_loader.read_data()
    #data_new = []
    # predicted_answer = []
    # target_answer = []
    data_list = []
    for idx, batch in zip(tqdm(range(len(test_gen)), desc='Test batch: ...'), test_gen):
        input_ids = batch[0].pop('input_ids').to(device)
        pixel_values = batch[0].pop('pixel_values').to(device)
        attention_masked = batch[0].pop('attention_mask').to(device)
        image_paths = batch[1]
        questions = batch[2]
        answers = batch[3]
        out = model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_masked)

        #generated_texts_batch = []
        # Iterate over each element in the batch

        for i in range(out.size(0)):
            single_element = []
            token_ids = out[i]
            generated_text = processor.decode(token_ids, skip_special_tokens=True)
            single_element.append(image_paths[i])
            single_element.append(questions[i])
            single_element.append(answers[i])
            single_element.append(generated_text)
            data_list.append(single_element)
    df = pd.DataFrame(data_list, columns=['image_path', 'question', 'target_answer', 'predicted_answer'])
    df.to_excel(generated_result_excel_file, index=False)
    #print(df)