import os
import json
import random
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from transform.randaugment import RandomAugment
from torch.utils.data import Dataset
import re
from ruamel.yaml import YAML
from torchvision.datasets.utils import download_url
#from data.utils import pre_question


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    )
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root

        if split == 'train':
            urls = {'vqa_train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}

            self.annotation = []
            for f in train_files:
                download_url(urls[f], ann_root)
                self.annotation += json.load(open(os.path.join(ann_root, '%s.json' % f), 'r'))
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json', ann_root)
            self.annotation = json.load(open(os.path.join(ann_root, 'vqa_test.json'), 'r'))

            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',
                         ann_root)
            self.answer_list = json.load(open(os.path.join(ann_root, 'answer_list.json'), 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.vqa_root, ann['image'])
        elif ann['dataset'] == 'vg':
            image_path = os.path.join(self.vg_root, ann['image'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'])
            question_id = ann['question_id']
            return image, question, question_id


        elif self.split == 'train':

            question = pre_question(ann['question'])

            if ann['dataset'] == 'vqa':
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann['answer'])
                    else:
                        answer_weight[answer] = 1 / len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset'] == 'vg':
                answers = [ann['answer']]
                weights = [0.2]

            return image, question, answers, weights

def create_dataset( config, min_scale=0.5):
    """

    :param config:
    :param min_scale:
    :return:
    """
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            # RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
            #                                   'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'],
                            train_files=config['train_files'], split='train')
    test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
    return train_dataset, test_dataset
