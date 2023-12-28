# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import os
import torch
import numpy as np
from utils import logging
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import tempfile


def modify_lr_strategy(cfg, current_epoch):
    # Set up the learning rate for current epoch
    if cfg.TRAIN.LR_scheduler == 'MilestonesLR':
        milestone_lists = [cfg.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES, cfg.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES]
        init_lr_list = [cfg.TRAIN.ENCODER_LEARNING_RATE, cfg.TRAIN.DECODER_LEARNING_RATE]
        milestone_lists.append(cfg.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES)
        init_lr_list.append(cfg.TRAIN.MERGER_LEARNING_RATE)
        current_milestone_list = []
        current_epoch_lr_list = []
        for milestones, init_lr in zip(milestone_lists, init_lr_list):
            milestones = np.array(milestones) - current_epoch
            init_lr = init_lr * cfg.TRAIN.MILESTONESLR.GAMMA ** len(np.where(milestones <= 0)[0])
            milestones = list(milestones[len(np.where(milestones <= 0)[0]):])
            current_milestone_list.append(milestones)
            current_epoch_lr_list.append(init_lr)
        cfg.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES = current_milestone_list[0]
        cfg.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES = current_milestone_list[1]
        cfg.TRAIN.ENCODER_LEARNING_RATE = current_epoch_lr_list[0]
        cfg.TRAIN.DECODER_LEARNING_RATE = current_epoch_lr_list[1]
        cfg.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES = current_milestone_list[2]
        cfg.TRAIN.MERGER_LEARNING_RATE = current_epoch_lr_list[2]
    else:
        raise ValueError(f'{cfg.TRAIN.LR_scheduler} is not supported.')
    return cfg


def load_data(cfg):
    # Set up data augmentation 
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])

    # Set up data loader 
    train_dataset, _ = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms)
    val_dataset, val_file_num = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfg.CONST.BATCH_SIZE_PER_GPU, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    return train_data_loader, train_sampler, val_data_loader, val_file_num


def setup_network(cfg, encoder, decoder, merger):
    # Set up networks
    logging.info('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.info('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.info('Parameters in Merger: %d.' % (utils.helpers.count_parameters(merger)))

    # Initialize weights of networks
    decoder.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)

    # set sync bn
    if cfg.TRAIN.SYNC_BN:
        if torch.distributed.get_rank() == 0:
            print('Setting sync_batchnorm ...')
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
        merger = torch.nn.SyncBatchNorm.convert_sync_batchnorm(merger)
    else:
        if torch.distributed.get_rank() == 0:
            print('Without sync_batchnorm')
    
    device = torch.cuda.current_device()
    
    # distributed data parallel
    encoder = torch.nn.parallel.DistributedDataParallel(
        encoder.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder.to(device), device_ids=[device], output_device=device)
    merger = torch.nn.parallel.DistributedDataParallel(merger.to(device), find_unused_parameters=True, device_ids=[device], output_device=device)
    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    
    if cfg.TRAIN.RESUME_TRAIN and 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
        init_epoch = checkpoint['epoch_idx'] + 1
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        merger.load_state_dict(checkpoint['merger_state_dict'])
        
        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))
        
        # resume the learning-rate strategy
        cfg = modify_lr_strategy(cfg, init_epoch)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weights.pth')
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'merger_state_dict': merger.state_dict()
        }
        
        if torch.distributed.get_rank() == 0:
            torch.save(checkpoint, checkpoint_path)
        torch.distributed.barrier()
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        merger.load_state_dict(checkpoint['merger_state_dict'])
        
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)

    return init_epoch, best_iou, best_epoch, encoder, decoder, merger, cfg


def setup_writer(cfg):
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
    return train_writer, val_writer


def solver(cfg, encoder, decoder, merger):
    encoder_solver = torch.optim.AdamW(encoder.parameters(),
                                       lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        
    decoder_solver = torch.optim.AdamW(decoder.parameters(),
                                       lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)

    merger_solver = torch.optim.AdamW(merger.parameters(),
                                        lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    return encoder_solver, decoder_solver, merger_solver
