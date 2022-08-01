import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import numpy as np

from torchinfo import summary

from ..storage import DataStore
from ..util import to_float
from ..config import CfgNode
from ..progress_bar import progressbar
from ..model import build_model
from ..dataset import build_loaders
from ..loss import build_loss
from ..metrics import build_metrics
from ..plotter import build_plotter
from ..optim import build_optim
from ..sched import build_sched


class Trainer:

    def __init__(self, config: CfgNode):
        self.model = build_model(config)
        if config.debug_mode:
            print(self.model)
            print(summary(self.model, (1, config.data.images.stack, 256, 256), device='cpu'))
        self.train_dl, self.valid_dl = self.build_loaders(config)
        self.test_dl = None
        self.should_test = False
        self.loss_func_t = build_loss(config)
        self.metrics = build_metrics(config)
        self.plotter = build_plotter(self)
        self.opt_t, self.opt_kws = build_optim(config)
        self.sched_t, self.sched_kws = build_sched(config, len(self.train_dl))

        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump())

        self.n_epochs = config.training.n_epochs
        self.device = torch.device(config.training.device)
        self.output_dir = config.output_dir
        self.plot_every = config.training.plot_every
        self.checkpoint_every = config.training.checkpoint_every
        self.validate_every = config.training.validate_every
        self.i = self.tbn = self.vbn = self.total_valid_loss = self.total_train_loss = self.total_test_loss = 0
        self.min_valid_loss = np.inf
        self.bar = self.last_checkpoint = None

        # used by sub_classes
        self.prefix = ''

    def build_loaders(self, config):
        return build_loaders(config)

    @property
    def should_plot(self) -> bool:
        return (self.i % self.plot_every == 0) or (self.i == self.n_epochs - 1)

    @property
    def should_validate(self) -> bool:
        return (self.i % self.validate_every == 0) or self.should_plot or self.validate_every == 1

    @property
    def should_checkpoint(self) -> bool:
        return self.i % self.checkpoint_every == 0

    @staticmethod
    def init_store_metadata(store):
        store.add_metadata_re('loss.*', logy=True)
        store.add_metadata('learning_rate', xlabel='Batch [#]')
        store.add_metadata_re(r'.*\.per_batch\..*', xlabel='Training Batch [#]')
        store.add_metadata_re(r'.*\.per_batch\.valid', xlabel='Valid Batch [#]')
        store.add_metadata_re(r'.*\.per_epoch\..*', xlabel='Epoch [#]')
        store.add_metadata_re(r'metrics\..*', xlabel='Epoch [#]')
        store.add_metadata_re(r'metrics\.by_class\..*', plot_timeseries=False)

    def do_train(self, store, opt, scheduler, loss_func):
        for batch in self.train_dl:
            inp = batch['inputs'].to(self.device)
            tgt = batch['targets'].to(self.device)
            opt.zero_grad()
            out = self.model(inp)

            loss = loss_func(out, tgt)

            train_loss = loss.item()
            self.total_train_loss += train_loss
            loss.backward()
            opt.step()
            scheduler.step()
            store.add_scalar('learning_rate', self.tbn, scheduler.get_last_lr()[0])
            self.tbn += 1
            store.add_scalar('loss.per_batch.train', self.tbn, train_loss / len(batch))

        store.add_scalar('loss.per_epoch.train', self.i, self.total_train_loss / len(self.train_dl))

    def validate_or_test(self, dataloader, loss_func, store, is_test: bool):

        # axes = self.plot_prep()

        metrics = defaultdict(list)
        valid_or_test = 'test' if is_test else 'valid'

        if self.should_plot:
            self.plotter.init()

        with torch.no_grad():
            for batch in dataloader:
                inp = batch['inputs'].to(self.device)
                tgt = batch['targets'].to(self.device)
                tags = batch['tag']
                out = self.model(inp)

                loss = loss_func(out, tgt)
                _loss = loss.item()
                if is_test:
                    self.total_test_loss += _loss
                else:
                    self.total_valid_loss += _loss
                    self.vbn += 1

                store.add_scalar(f'loss.per_batch.{valid_or_test}', self.vbn, _loss / len(batch))

                for metric_name, metric_func in self.metrics.items():
                    for o, t, tag in zip(out, tgt, tags):
                        metric_value = metric_func(o, t)
                        try:
                            metric_value = to_float(metric_value)
                        except TypeError:
                            print(metric_name)
                            raise
                        metrics[f'metrics.{valid_or_test}.{metric_name}'].append(metric_value)
                        metrics[f'metrics.by_class.{tag}.{valid_or_test}.{metric_name}'].append(metric_value)

                # self.plot_valid_batch(axes, tgt, out)
                if self.should_plot:
                    self.plotter.plot_targets(batch, out)

            for key, values in metrics.items():
                store.add_scalar(key, self.i, np.mean(values))
            # self.plot_finalise(axes, is_test=is_test)
            if self.should_plot:
                self.plotter.finalise()
        store.add_scalar(f'loss.per_epoch.{valid_or_test}', self.i,
                         (self.total_test_loss if is_test else self.total_valid_loss) / len(dataloader))

    def do_validation(self, store, loss_func):
        self.validate_or_test(self.valid_dl, loss_func, store=store, is_test=False)

    def do_test(self, store, loss_func):
        assert self.should_test
        assert self.test_dl is not None
        self.validate_or_test(self.test_dl, loss_func, store=store, is_test=True)

    def save_dataset_contents(self, dataloader, tag):
        sources = []
        for d in dataloader:
            batch_sources = np.array(d['sources'], dtype=str).flatten()
            sources.extend([os.path.normpath(s) for s in batch_sources])

        with open(f'{self.output_dir}/{self.prefix}dataset_{tag}_listing.txt', 'w') as f:
            for line in sources:
                f.write(f'{line}\n')

    def train(self):
        self.i = self.tbn = self.vbn = self.total_valid_loss = self.total_train_loss = 0
        self.min_valid_loss = np.inf

        self.device = torch.device(self.device)
        print(f'Running on {self.device}')
        self.model = self.model.to(self.device)

        with open(f'{self.output_dir}/model.txt', 'w') as f:
            f.write(str(self.model))

        self.save_dataset_contents(self.train_dl, 'train')
        self.save_dataset_contents(self.valid_dl, 'valid')

        opt = self.opt_t(self.model.parameters(), **self.opt_kws)
        scheduler = self.sched_t(opt, **self.sched_kws)
        loss_func = self.loss_func_t()
        self.bar = progressbar(range(self.n_epochs), unit='epoch')
        with DataStore(self.output_dir, prefix=self.prefix) as store:
            self.init_store_metadata(store)
            self.do_validation(store, loss_func)
            for _i in self.bar:
                self.i = _i + 1
                self.total_train_loss = 0
                self.do_train(store, opt, scheduler, loss_func)
                if self.should_validate:
                    self.total_valid_loss = 0.0
                    self.do_validation(store, loss_func)

                if self.should_plot:
                    store.plot()

                if self.should_checkpoint:
                    self.checkpoint()
                    self.last_checkpoint = self.i

                if self.total_valid_loss / len(self.valid_dl) < self.min_valid_loss:
                    self.min_valid_loss = self.total_valid_loss / len(self.valid_dl)
                    torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_min_loss.pth')

                self.update_progress()
            self.checkpoint()
            if self.should_test:
                assert self.test_dl
                print('Loading best model state (decided based on validation set results)')
                self.model.load_state_dict(torch.load(f'{self.output_dir}/{self.prefix}model_min_loss.pth'))
                print('Running on test dataset')
                self.do_test(store, loss_func)
            return store.get_data()

    def checkpoint(self):
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_sate_final.pth')
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.prefix}model_sate_at_epoch={self.i}.pth')

    def update_progress(self):
        desc = f'{self.prefix}t:{self.total_train_loss / len(self.train_dl):.2e}|v:{self.total_valid_loss / len(self.valid_dl):.2e}|'
        if self.last_checkpoint is None:
            desc += '!*'
        else:
            desc += f'c:{self.last_checkpoint}'
            if self.i > self.last_checkpoint:
                desc += '*'
        self.bar.set_description(desc, False)