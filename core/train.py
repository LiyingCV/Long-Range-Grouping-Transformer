# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers
from utils import logging

from time import time

import core.pipeline_train as pipeline
from core.test import test_net, batch_test

from models.encoder.encoder import Encoder 
from models.merger.merger import Merger 
from models.decoder.decoder import Decoder

from losses.losses import DiceLoss
from utils.average_meter import AverageMeter


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    train_data_loader, train_sampler, val_data_loader, val_file_num = pipeline.load_data(cfg)

    # load model
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)
    init_epoch, best_iou, best_epoch, encoder, decoder, merger, cfg = \
        pipeline.setup_network(cfg, encoder, decoder, merger)

    # Set up solver
    encoder_solver, decoder_solver, merger_solver = pipeline.solver(cfg, encoder, decoder, merger)

    # Set up learning rate scheduler to decay learning rates dynamically
    if cfg.TRAIN.LR_scheduler == 'MilestonesLR':
        encoder_lr_scheduler = \
            torch.optim.lr_scheduler.MultiStepLR(
                encoder_solver, milestones=[lr for lr in cfg.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES],
                gamma=cfg.TRAIN.MILESTONESLR.GAMMA)
        decoder_lr_scheduler = \
            torch.optim.lr_scheduler.MultiStepLR(
                decoder_solver, milestones=[lr for lr in cfg.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES],
                gamma=cfg.TRAIN.MILESTONESLR.GAMMA)
        merger_lr_scheduler = \
            torch.optim.lr_scheduler.MultiStepLR(
                merger_solver, milestones=[lr for lr in cfg.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES],
                gamma=cfg.TRAIN.MILESTONESLR.GAMMA)
    else:
        raise ValueError(f'{cfg.TRAIN.LR_scheduler} is not supported.')

    # Set up loss functions
    loss_function = DiceLoss()

    # Summary writer for TensorBoard
    val_writer = None
    if torch.distributed.get_rank() == 0:
        train_writer, val_writer = pipeline.setup_writer(cfg)
    torch.distributed.barrier()

    # Training loop
    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        train_sampler.set_epoch(epoch_idx)  

        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        if torch.distributed.get_rank() == 0:
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) \
                in enumerate(train_data_loader):
            # Measure data time
            rendering_images = rendering_images[:, :n_views_rendering, ::]

            if torch.distributed.get_rank() == 0:
                data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = \
                utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volumes = \
                utils.helpers.var_or_cuda(ground_truth_volumes).to(torch.cuda.current_device())

            # Train the encoder, decoder, and merger
            # encoder
            image_features = encoder(rendering_images)

            # merger
            context = merger(image_features)

            # decoder
            generated_volumes = decoder(context).squeeze(dim=1)

            # Loss
            loss = loss_function(generated_volumes, ground_truth_volumes)

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            merger.zero_grad()

            loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            merger_solver.step()

            loss = utils.helpers.reduce_value(loss)

            # logging (only on main processing)
            if torch.distributed.get_rank() == 0:
                # Append loss to average metrics
                losses.update(loss.item())

                # Append loss to TensorBoard
                n_itr = epoch_idx * n_batches + batch_idx
                train_writer.add_scalar('BatchLoss', loss.item(), n_itr)

                # Tick / tock
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
                    logging.info(
                        '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f'
                        % (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1,
                           n_batches, batch_time.val, data_time.val, loss.item()))

                    print('LearningRate:\tencoder: %f | decoder: %f | merger: %f | %d_views_rendering' %
                              (encoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                               decoder_lr_scheduler.optimizer.param_groups[0]['lr'],
                               merger_lr_scheduler.optimizer.param_groups[0]['lr'], n_views_rendering))
                else:
                    logging.debug(
                        '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f'
                        % (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                           loss.item()))

        torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        if torch.distributed.get_rank() == 0:
            train_writer.add_scalar('EpochLoss', losses.avg, epoch_idx + 1)

            # Tick / tock
            epoch_end_time = time()
            logging.info('[Epoch %d/%d] EpochTime = %.3f (s) Loss = %.4f' %
                         (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, losses.avg))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, val_data_loader, val_file_num, val_writer, encoder, decoder, merger)

        # Save weights to file  #保存模型
        if torch.distributed.get_rank() == 0:
            if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
                file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
                if iou > best_iou:
                    best_iou = iou
                    best_epoch = epoch_idx
                    file_name = 'checkpoint-best.pth'

                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                if not os.path.exists(cfg.DIR.CHECKPOINTS):
                    os.makedirs(cfg.DIR.CHECKPOINTS)

                checkpoint = {
                    'epoch_idx': epoch_idx,
                    'best_iou': best_iou,
                    'best_epoch': best_epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'merger_state_dict': merger.state_dict()
                }

                torch.save(checkpoint, output_path)
                logging.info('Saved checkpoint to %s ...' % output_path)

    # Close SummaryWriter for TensorBoard
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        train_writer.close()
        val_writer.close()
        torch.distributed.destroy_process_group()
