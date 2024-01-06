# -*- coding: utf-8 -*-

from easydict import EasyDict as edict

__C = edict()
cfg = __C


# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = '/PATH/TO/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/PATH/TO/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'

# Dataset
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet'  
__C.DATASET.TEST_DATASET = 'ShapeNet'  

# Common
__C.CONST = edict()
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.BATCH_SIZE_PER_GPU = 1
__C.CONST.N_VIEWS_RENDERING = 3
__C.CONST.NUM_WORKER = 20  # number of data workers
__C.CONST.WEIGHTS = \
    '/PATH/TO/checkpoint.pth'

# Directories
__C.DIR = edict()
__C.DIR.OUT_PATH = '/PATH/TO/output/'

# Network
__C.NETWORK = edict()
__C.NETWORK.ENCODER = edict()
# vit
__C.NETWORK.ENCODER.VIT = edict()
__C.NETWORK.ENCODER.VIT.MODEL_NAME = 'vit_deit_base_distilled_patch16_224' 
__C.NETWORK.ENCODER.VIT.PRETRAINED = True
# Decoder
__C.NETWORK.DECODER = edict()
__C.NETWORK.DECODER.VOXEL_SIZE = 32
__C.NETWORK.DECODER.GROUP = edict()
__C.NETWORK.DECODER.GROUP.SOFTMAX_DROPOUT = 0.4  
__C.NETWORK.DECODER.GROUP.ATTENTION_MLP_DROPOUT = 0.
__C.NETWORK.DECODER.GROUP.DEPTH = 8
__C.NETWORK.DECODER.GROUP.HEADS = 12 
__C.NETWORK.DECODER.GROUP.DIM = 768
# Merger (STM)
__C.NETWORK.MERGER = edict()
__C.NETWORK.MERGER.STM = edict()
__C.NETWORK.MERGER.STM.DIM = 768
__C.NETWORK.MERGER.STM.OUT_TOKEN_LENS = [196, 196]
__C.NETWORK.MERGER.STM.K = 15
__C.NETWORK.MERGER.STM.NUM_HEAD = 12

# Training
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False 
__C.TRAIN.SYNC_BN = True  # Distributed training need SYNC_BN is True
__C.TRAIN.NUM_EPOCHS = 110
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]

__C.TRAIN.ENCODER_LEARNING_RATE = 1e-4
__C.TRAIN.DECODER_LEARNING_RATE = 1e-4
__C.TRAIN.MERGER_LEARNING_RATE = 1e-4

# for MilestonesLR
__C.TRAIN.LR_scheduler = 'MilestonesLR' 
__C.TRAIN.MILESTONESLR = edict()
__C.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES = [60, 90]
__C.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES = [60, 90]
__C.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES = [60, 90]
__C.TRAIN.MILESTONESLR.GAMMA = .1

__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.SAVE_FREQ = 10  # weights will be overwritten every save_freq epoch
__C.TRAIN.SHOW_TRAIN_STATE = 500
__C.TRAIN.TEST_AFTER_TRAIN = True

# Testing options
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.3, .4, .5]