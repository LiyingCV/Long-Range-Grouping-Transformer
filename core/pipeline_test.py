# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>
import torch
import os
import numpy as np
import json
from utils import logging

import utils.data_loaders
import utils.data_transforms
import utils.helpers


def load_data(cfg, test_data_loader=None, test_file_num=None):
    # Load taxonomies of dataset
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}
    
    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ToTensor(),
            utils.data_transforms.normalize
        ])
        
        dataset, test_file_num = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg).get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=test_sampler,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True)
    return taxonomies, test_data_loader, test_file_num


def setup_network(cfg, encoder, decoder, merger):
    device = torch.cuda.current_device()

    encoder = torch.nn.parallel.DistributedDataParallel(encoder.cuda(), device_ids=[device], output_device=device)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder.cuda(), device_ids=[device], output_device=device)
    merger = torch.nn.parallel.DistributedDataParallel(merger.cuda(), device_ids=[device], output_device=device)
    
    logging.info('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device(device))
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    merger.load_state_dict(checkpoint['merger_state_dict'])

    return encoder, decoder, merger, epoch_idx


def combine_test_iou(test_iou, taxonomies_list, taxonomies, test_file_num):
    world_size = int(os.environ['WORLD_SIZE'])
    all_test_iou = [torch.zeros_like(test_iou) for _ in range(world_size)]
    all_taxonomies_list = [torch.zeros_like(taxonomies_list) for _ in range(world_size)]
    torch.distributed.all_gather(all_test_iou, test_iou)
    torch.distributed.all_gather(all_taxonomies_list, taxonomies_list)
    if torch.distributed.get_rank() == 0:
        redundancy = test_file_num % world_size
        if redundancy == 0:
            redundancy = world_size
        for i in range(world_size):
            all_test_iou[i] = all_test_iou[i] \
                if i < redundancy else all_test_iou[i][:-1, :]
            all_taxonomies_list[i] = all_taxonomies_list[i] \
                if i < redundancy else all_taxonomies_list[i][:-1]
        all_test_iou = torch.cat(all_test_iou, dim=0).cpu().numpy()  # [sample_num, 4]
        all_taxonomies_list = torch.cat(all_taxonomies_list).cpu().numpy()  # [sample_num]
        test_iou = dict()
        for taxonomy_id, sample_iou in zip(all_taxonomies_list, all_test_iou):
            if taxonomies[taxonomy_id] not in test_iou.keys():
                test_iou[taxonomies[taxonomy_id]] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomies[taxonomy_id]]['n_samples'] += 1
            test_iou[taxonomies[taxonomy_id]]['iou'].append(sample_iou)
        return test_iou
    else:
        return


def output(cfg, test_iou, taxonomies):
    mean_iou = []
    n_samples = 0
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        n_samples += test_iou[taxonomy_id]['n_samples']
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('\n')
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        print('N/a', end='\t\t')
        
        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()

    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    return mean_iou


