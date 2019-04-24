import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, g_optimizer, d_optimizer, resume, config, train_recorder=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        #  self.device, device_ids = self._prepare_device(config['n_gpu'])
        #  self.model = model.to(self.device)
        #  if len(device_ids) > 1:
            #  self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.device = config['device']

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_recorder = train_recorder

        if self.device is not None:
            self.model = model.cuda()
            self.loss = loss.cuda()

        self.trainer_config = config['trainer']['args']
        self.use_summaryWriter = config['use_summaryWriter']
        self.batch_size = self.config['dataloader']['args']['batch_size']
        self.epochs = self.trainer_config['epochs']
        self.save_period = self.trainer_config['save_period']

        #  self.verbosity = cfg_trainer['verbosity']
        #  self.monitor = cfg_trainer.get('monitor', 'off')
        #
        #  # configuration to monitor model performance and save best
        #  if self.monitor == 'off':
        #      self.mnt_mode = 'off'
        #      self.mnt_best = 0
        #  else:
        #      self.mnt_mode, self.mnt_metric = self.monitor.split()
        #      assert self.mnt_mode in ['min', 'max']
        #
        #      self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
        #      self.early_stop = cfg_trainer.get('early_stop', math.inf)
        
        self.start_epoch = 1
        self.global_step = 0

        if self.use_summaryWriter:
            self.writer = SummaryWriter(self.trainer_config['log_dir']) # tensorboard 建立的是目录，它会自动产生文件名，不需要手动指定

        if resume:
            self._resume_checkpoint(resume)
    
    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """ 
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_result = {}
            train_result = self._train_epoch(epoch)
            
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            val_result = {}
            if self.trainer_config['do_validation'] and self.do_validation:
                self.logger.info('doing validation ... ')
                val_result = self._valid_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch, **train_result, **val_result}
            if self.train_recorder is not None:
                self.train_recorder.add_entry(log)

            self.logger.info(log)

            # 保存一个最佳的验证集结果，每次与它比较，效果更好则best=True，否则best=False
            best = False
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_name = type(self.model).__name__
        state = {
            'model_name': model_name,
            'epoch': epoch,
            'global_step': self.global_step, 
            'recorder': self.train_recorder,
            'state_dict': self.model.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            #  'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.trainer_config['save_dir'], 'model_{}_epoch{}.pth'.format(model_name, epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.trainer_config['save_dir'], 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc:storage) # load parameters to CPU
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step'] + 1
        #  self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model_name'] != self.config['model_name']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    
        self.train_recorder = checkpoint['recorder']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
