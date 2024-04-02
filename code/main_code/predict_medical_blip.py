import pandas as pd
import os
from torch.utils import data
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from ruamel.yaml import YAML
import torch
from tqdm import tqdm
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
#from nltk.translate.rouge_score import rouge_n, rouge_l
#from rouge import Rouge
#from rouge import Rouge
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from sklearn.metrics import jaccard_score
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')


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
xdf_dset_test = pd.read_excel(test_data_excel_file)#.head(10)

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
    def __init__(self,config,processor):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.processor = processor
        # self.model = model
    def read_data(self):
        list_of_ids_test = list(xdf_dset_test.index)
        partition = {
            'test': list_of_ids_test
        }
        params = {'batch_size': self.BATCH_SIZE, 'shuffle': False}
        test_set = CustomDataset(partition['test'], self.processor )
        test_generator = data.DataLoader(test_set, **params)
        return test_generator
def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    bleu, rouge, jaccard, exact match, f1 score, meteo
    list of metrics: bleu, rouge, jaccard, exact match, f1 score, meteo
    list of aggregates : avg, sum
    :return:
    '''

    def bleu_score(y_true, y_pred):
        bleu_scores = np.zeros(len(y_true))
        for i in range(len(y_true)):
            bleu_scores[i] = nltk.translate.bleu_score.sentence_bleu([y_true[i]], y_pred[i], smoothing_function=SmoothingFunction().method2)

        return np.mean(bleu_scores)

    def rouge_score(y_true, y_pred):
        rougeL_scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        for y_pred, y_true in zip(y_pred, y_true):
            scores = scorer.score(y_pred, y_true)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

        return  avg_rougeL
    def compute_meteor_score(y_true_list, y_pred_list):
        scores = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            tokens_true = word_tokenize(y_true)
            tokens_pred = word_tokenize(y_pred)
            score = meteor_score([tokens_true], tokens_pred)
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        return avg_score
    def jaccard_similarity(y_true, y_pred):
        intersection = len(set(y_true) & set(y_pred))
        union = len(set(y_true) | set(y_pred))
        return intersection / union if union else 0.0
    def exact_match(y_true, y_pred):
        num_exact_matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return num_exact_matches / len(y_true)
    def f1_score(y_true, y_pred):
        reference_tokens = set(y_true.split())
        hypothesis_tokens = set(y_pred.split())
        if len(hypothesis_tokens) == 0:
            return 0.0
        precision = len(reference_tokens.intersection(hypothesis_tokens)) / len(hypothesis_tokens)
        recall = len(reference_tokens.intersection(hypothesis_tokens)) / len(reference_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def calculate_f1_scores(y_true_list, y_pred_list):
        f1_scores = []
        for y_true, y_pred in zip(y_true_list, y_pred_list):
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        average_f1_score = sum(f1_scores) / len(f1_scores)
        return average_f1_score

    xcont = 1
    xsum = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'bleu':
            # f1 score average = micro
            xmet = bleu_score(y_true, y_pred)
        elif xm == 'rouge':
            # f1 score average = macro
            xmet = rouge_score(y_true, y_pred)
        elif xm == 'meteor':
            # f1 score average =
            xmet = compute_meteor_score(y_true, y_pred)
        elif xm == 'jac':
             # Cohen kappa
            xmet = jaccard_similarity(y_true, y_pred)
        elif xm == 'em':
            # Accuracy
            xmet = exact_match(y_true, y_pred)
        elif xm == 'f1':
            # Matthews
            xmet = calculate_f1_scores(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict
def model_definition():
    model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")
    model.to(device)
    return model
def eval_model(test_gen,processor, list_of_metrics, list_of_agg):
    model = model_definition()
    data_list = []
    for idx, batch in zip(tqdm(range(len(test_gen)), desc='Test batch: ...'), test_gen):
        input_ids = batch[0].pop('input_ids').to(device)
        pixel_values = batch[0].pop('pixel_values').to(device)
        attention_masked = batch[0].pop('attention_mask').to(device)
        image_paths = batch[1]
        questions = batch[2]
        answers = batch[3]
        out = model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_masked)

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
    #df.to_excel(generated_result_excel_file, index=False)
    print(f"saved to excel")


    test_metrics = metrics_func(list_of_metrics, list_of_agg, df['target_answer'].values, df['predicted_answer'].values)
    xstrres = ""
    for met, dat in test_metrics.items():
        xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
    print(xstrres)



if __name__ == '__main__':

    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    test_loader = CustomDataLoader(config,processor)
    test_gen = test_loader.read_data()

    list_of_metrics = ['bleu', 'rouge', 'jac','em','f1','meteor']
    list_of_agg = ['avg', 'sum']
    eval_model(test_gen, processor, list_of_metrics, list_of_agg)
