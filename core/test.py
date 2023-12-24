# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import numpy as np
from utils import logging
from datetime import datetime as dt
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import core.pipeline_test as pipeline

from models.encoder.encoder import Encoder 
from models.merger.merger import Merger 
from models.decoder.decoder import Decoder

from losses.losses import DiceLoss
from utils.average_meter import AverageMeter


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_file_num=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Set up networks
    if decoder is None or encoder is None or Merger is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        merger = Merger(cfg)

        encoder, decoder, merger, epoch_idx = \
            pipeline.setup_network(cfg, encoder, decoder, merger)

    # Set up loss functions
    loss_function = DiceLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    taxonomies_list = []
    losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()

    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the encoder, decoder and merger
            # encoder
            image_features = encoder(rendering_images)

            # merger
            context = merger(image_features)

            # decoder
            generated_volume = decoder(context).squeeze(dim=1)
            generated_volume = generated_volume.clamp_max(1)

            # Loss
            loss = loss_function(generated_volume, ground_truth_volume)

            # Append loss and accuracy to average metrics
            loss = utils.helpers.reduce_value(loss)
            losses.update(loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            if torch.distributed.get_rank() == 0:
                # Print sample loss and IoU
                if (sample_idx + 1) % 50 == 0:
                    for_tqdm.update(50)
                    for_tqdm.set_description('Test[%d/%d] Taxonomy = %s Loss = %.4f' %
                                             (sample_idx + 1, n_samples, taxonomy_id, losses.avg))

                logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f IoU = %s' %
                              (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                               loss.item(), ['%.4f' % si for si in sample_iou]))

    test_iou = torch.cat(test_iou, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou = pipeline.combine_test_iou(test_iou, taxonomies_list, list(taxonomies.keys()), test_file_num)

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if torch.distributed.get_rank() == 0:
        # Output testing results
        mean_iou = pipeline.output(cfg, test_iou, taxonomies)

        # Add testing results to TensorBoard
        max_iou = np.max(mean_iou)
        if test_writer is not None:
            test_writer.add_scalar('EpochLoss', losses.avg, epoch_idx)
            test_writer.add_scalar('IoU', max_iou, epoch_idx)

        print('The IoU score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou))

        return max_iou


