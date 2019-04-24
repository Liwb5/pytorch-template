# coding:utf-8
import os
import sys 
import logging
import json
import random 
import argparse
import pickle
import datetime
import numpy as np
from pprint import pprint, pformat

import torch
from torch.utils.data import DataLoader, Dataset
import models.loss as module_loss
import models.metric as module_metric
import models

import dataloader as module_data
from trainer import Trainer
from utils import Recorder 
from utils import StreamToLogger 
from utils.config import *


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config, resume):
    set_seed(config['seed'])

    train_recorder = Recorder()

    # setup data_loader instances
    train_data = getattr(module_data, config['dataloader']['type'])(
                        data_path = config['dataloader']['args']['train_data'], 
                        data_quota = config['dataloader']['args']['data_quota']
                        )
    logging.info('using %d examples to train. ' % len(train_data))
    data_loader = DataLoader(dataset = train_data,
                            batch_size = config['dataloader']['args']['batch_size'])

    #  val_data = getattr(module_data, config['dataloader']['type'])(
    #                      data_path = config['dataloader']['args']['val_data'],
    #                      data_quota = config['dataloader']['args']['data_quota']
    #                      )
    #  logging.info('using %d examples to val. ' % len(val_data))
    #  valid_data_loader = DataLoader(dataset = val_data,
    #                          batch_size = config['data_loader']['batch_size'])

    # build model architecture
    model = getattr(models, config['model']['type'])(config['model']['args'], device=config['device'])
    
    logging.info(['my PID is: ', os.getpid()])
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])()

    #  metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    g_trainable_params = filter(lambda p: p.requires_grad, model.G.parameters())
    g_optimizer = getattr(torch.optim, config['optimizer']['generator']['type'])(g_trainable_params, **config['optimizer']['generator']['args'])

    d_trainable_params = filter(lambda p: p.requires_grad, model.D.parameters())
    d_optimizer = getattr(torch.optim, config['optimizer']['discriminator']['type'])(d_trainable_params, **config['optimizer']['discriminator']['args'])

    trainer = Trainer(model, loss,  g_optimizer, d_optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      metrics=None,
                      lr_scheduler=None,
                      train_recorder=train_recorder)

    logging.info('begin training. ')
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to checkpoint that you want to reload (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPU to enable (default: None)')
    parser.add_argument('-n', '--name', default=None, type=str,
                           help='task name (default: None)')
    args = parser.parse_args()

    if args.config:
        config = get_config_from_yaml(args.config)
        config = process_config(config)
        # save config file so that we can know the config when we look back
        save_config(args.config, config['trainer']['args']['save_dir']) 
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume, map_location=lambda storage, loc:storage)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c configs/config.yaml', for example.")

    log_format='%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    logging.basicConfig(filename = ''.join((config['trainer']['args']['log_dir'], 'log')),
                        filemode = 'a',
                        level = getattr(logging, config['log_level'].upper()),
                        format = log_format)

    # redirect stderr to logging file
    stderr_logger = logging.getLogger('stderr')
    sys.stderr = StreamToLogger(stderr_logger,getattr(logging, config['log_level'].upper()))

    if args.device is not None:
        logging.info('using GPU device %s'%(args.device))
        torch.cuda.set_device(int(args.device))
        config['device'] = int(args.device)
    else:
        config['device'] = args.device

    main(config, args.resume)
