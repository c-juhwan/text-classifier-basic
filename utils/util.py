import os
import time
import random
import torch
import torch.nn as nn
import numpy as np

def set_random_seed(seed: int):
    """Set random seed for all randomness sources
    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_tb_experiment_name(args, ts):
    """Get experiment name for tensorboard visualization
    Args:
        args (Namespace): arguments
        ts (int): timestamp
    Return:
        exp_name (str): experiment name
    """
    exp_name = str()

    exp_name += "ModelName=%s - " % (args.model_name + '_' + args.model_type)
    exp_name += "DatasetName=%s - " % (args.dataset_name)
    exp_name += "BS=%i_" % args.batch_size 
    if args.epoch is not None:
        exp_name += "EP=%i_" % args.epoch
    if args.learning_rate is not None:
        exp_name += "LR=%.4f_" % args.learning_rate
    exp_name += "SEED=%i_" % args.seed
    exp_name += "TS=%s" % ts

    return exp_name