# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2023-11-18
Version: 1.0
"""

"""
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
"""

import os
import json
import random
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
#from transform.randaugment import RandomAugment
from torch.utils.data import Dataset
import re
from ruamel.yaml import YAML
from blip_train import create_loader
from blip_train import create_dataset
from blip_train import vqa_collate_fn
CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)

#CODE_DIR = os.path.dirname(MAIN_CODE_DIR)
#os.chdir('..')
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
def main():
    """

    :rtype: object
    """
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "vqa.yml" )
    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    print("Creating vqa datasets")
    datasets = create_dataset(config)
    samplers = [None, None]
    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    # model related things need to be added
    for i, (image, question, answer, weights, n) in enumerate(train_loader):
        print(question[0])
        print(answer[0])
        print(weights[0])
        print(n[0])
        break


if __name__ == "__main__":
    main()
