import numpy as np
import torch
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
from base import BaseTrainer
from utils import inf_loop, MetricTracker, plot_confusion_matrix


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

# tile-level
class CrcTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        # for class weighting loss func
        self.class_to_tilenum = dict(Counter(self.data_loader.dataset.labels))
        self.pos_weight = torch.Tensor([self.class_to_tilenum[0] / self.class_to_tilenum[1]])
        # self.alpha = self.class_to_tilenum[0] / (self.class_to_tilenum[0] + self.class_to_tilenum[1])

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    # additional logging
    # - log metrics of best model at the end of training
    # - log only trainable parameters
    # - doesn't add input images to tensorboard writer
    # - adds confusion matrix of train and val to tensorboard writer every epoch
    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                if 'epoch' in key:
                    self.logger.info('    {:15s}: {}'.format(str(key), value))
                elif 'loss' in key:
                    self.logger.info('    {:15s}: {:.6f}'.format(str(key), value))
                else:
                    self.logger.info('    {:15s}: {:.4f}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    best_log = log
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)    
        
        if self.mnt_mode != 'off':
            self.logger.info("Final best: ")
            for key, value in best_log.items():
                if 'epoch' in key:
                    self.logger.info('    {:15s}: {}'.format(str(key), value))
                elif 'loss' in key:
                    self.logger.info('    {:15s}: {:.6f}'.format(str(key), value))
                else:
                    self.logger.info('    {:15s}: {:.4f}'.format(str(key), value))
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        # preds and labels for the entire train set after training 1 epoch
        tile_preds = []
        tile_probs = []
        tile_labels = []

        # for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data, target, path) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            pos_weight = self.pos_weight.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            target = target.unsqueeze(1).float()
            loss = self.criterion(output, target, pos_weight=pos_weight)
            loss.backward()
            self.optimizer.step()

            prob = torch.sigmoid(output)
            pred = torch.round(prob)
            tile_probs.extend(prob.detach().cpu().numpy())
            tile_preds.extend(pred.detach().cpu().numpy())
            tile_labels.extend(target.detach().cpu().numpy())

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        # add auroc to log and tensorboard every epoch (not per step)
        auroc = roc_auc_score(tile_labels, tile_probs)
        additional_log = {'auroc' : auroc}
        log.update(additional_log)
        self.writer.add_scalar('auroc', auroc, global_step=epoch)

        # add confusion matrix to tensorboard every epoch (not per step)
        # labels=[0, 1] is set for binary classification - change for multi-class classification
        cm = confusion_matrix(tile_labels, tile_preds, labels=[0, 1])
        cm_fig = plot_confusion_matrix(cm, [0, 1])
        self.writer.add_figure('confusion_matrix', cm_fig, global_step=epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        # preds and labels for entire val set after training 1 epoch
        tile_preds = []
        tile_probs = []
        tile_labels = []

        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            for batch_idx, (data, target, path) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                pos_weight = self.pos_weight.to(self.device)

                output = self.model(data)
                target = target.unsqueeze(1).float()
                loss = self.criterion(output, target, pos_weight=pos_weight)

                prob = torch.sigmoid(output)
                pred = torch.round(prob)
                tile_probs.extend(prob.detach().cpu().numpy())
                tile_preds.extend(pred.detach().cpu().numpy())
                tile_labels.extend(target.detach().cpu().numpy())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model's trainable parameters to the tensorboard
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()

        # add auroc to log and tensorboard every epoch (not per step)
        auroc = roc_auc_score(tile_labels, tile_probs)
        additional_log = {'auroc' : auroc}
        log.update(additional_log)
        self.writer.add_scalar('auroc', auroc, global_step=epoch)

        # add confusion matrix to tensorboard every epoch (not per step)
        # labels are set for binary classification
        cm = confusion_matrix(tile_labels, tile_preds, labels=[0, 1])
        cm_fig = plot_confusion_matrix(cm, [0, 1])
        self.writer.add_figure('confusion_matrix', cm_fig, global_step=epoch)

        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    