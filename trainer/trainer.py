import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, g_optimizer, d_optimizer, resume, config,
                 data_loader, metrics=None, valid_data_loader=None, lr_scheduler=None, train_recorder=None):
        super(Trainer, self).__init__(model, loss, metrics, g_optimizer, d_optimizer, resume, config, train_recorder)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_metrics = metrics is not None 

        if self.device is not None:
            self.fixed_noise = torch.randn(64, 100, 1, 1).cuda()
        else:
            self.fixed_noise = torch.randn(64, 100, 1, 1)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            #  self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        step_in_epoch = 0
        loss_g = 0
        loss_d = 0
        for step, dataset in enumerate(self.data_loader):
            self.global_step += 1
            step_in_epoch += 1

            # train discriminator   
            if self.device is not None:
                real_data = dataset.float().cuda()
                real_label = torch.ones(dataset.size()[0]).cuda()
                noise = torch.randn(dataset.size()[0], 100, 1, 1).cuda()
            else:
                real_data = dataset.float()
                real_label = torch.ones(dataset.size()[0])
                noise = torch.randn(dataset.size()[0], 100, 1, 1)

            fake_data = self.model.G(noise)
            if self.device is not None:
                fake_label = torch.zeros(fake_data.size()[0]).cuda()
            else:
                fake_label = torch.zeros(fake_data.size()[0])

            self.d_optimizer.zero_grad()
            real_output = self.model.D(real_data)
            real_loss = self.loss(real_output.squeeze(), real_label)
            real_loss.backward()

            fake_output = self.model.D(fake_data)
            fake_loss = self.loss(fake_output.squeeze(), fake_label)
            fake_loss.backward()
            loss_D = real_loss + fake_loss
            self.d_optimizer.step()

            #train generate
            if self.device is not None:
                fake_label = torch.ones(fake_data.size()[0]).cuda()
            else:
                fake_label = torch.ones(fake_data.size()[0])

            self.g_optimizer.zero_grad()
            fake_data = self.model.G(noise)
            fake_output = self.model.D(fake_data)
            loss_G = self.loss(fake_output.squeeze(), fake_label)
            loss_G.backward()
            self.g_optimizer.step()

            loss_g += loss_G.item()
            loss_d += loss_D.item() 

            if self.global_step % self.trainer_config['print_loss_every'] == 0:
                avg_loss_g = loss_g / self.trainer_config['print_loss_every']/self.batch_size
                avg_loss_d = loss_d / self.trainer_config['print_loss_every']/self.batch_size
                
                self.logger.info('Epoch: %d, global_batch: %d, Batch ID:%d g_loss:%f d_loss: %f '%(epoch, self.global_step, step_in_epoch, avg_loss_g, avg_loss_d))

                if self.use_summaryWriter:
                    self.writer.add_scalar('train/g_loss', avg_loss_g, self.global_step)
                    self.writer.add_scalar('train/d_loss', avg_loss_d, self.global_step)
                loss_g = 0
                loss_d = 0

        log = {}
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def compute_loss(self, logits, labels):
        pass
