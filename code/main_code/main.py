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
from blip_train import create_dataset, create_sampler
from blip_train import vqa_collate_fn

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
MODEL_DIR = CODE_DIR + os.sep + 'component' + os.sep + 'models'
import sys
sys.path.append(MODEL_DIR)
from blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule



#CODE_DIR = os.path.dirname(MAIN_CODE_DIR)
#os.chdir('..')
CONFIG_DIR = CODE_DIR + os.sep + 'configs'
EVALUATE =True
output_dir = CODE_DIR +os.sep+ 'output' + os.sep + 'VQA'
distributed = True

def main():
    """

    :rtype: object
    """
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_DIR + os.sep + "vqa.yml" )
    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    print("Creating vqa datasets")
    datasets = create_dataset(config)
    #samplers = [None, None]

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # models related things need to be added
    # for i, (image, question, answer, weights, n) in enumerate(train_loader):
    #     print(question[0])
    #     print(answer[0])
    #     print(weights[0])
    #     print(n[0])
    #     break

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    for epoch in range(0, config['max_epoch']):
        if not EVALUATE:
            if distributed:
               train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device)


        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

        dist.barrier()

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
