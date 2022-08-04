import os
from collections import defaultdict
from datetime import datetime
from traceback import format_exception

import torch
import numpy as np

from torchinfo import summary
from mldb import Database

from ..util import to_float
from ..config import CfgNode
from ..progress_bar import progressbar
from ..model import build_model
from ..dataset import build_dataloaders
from ..metrics import build_metrics
from ..optim import build_optim
from ..sched import build_sched
from .action_base import Action


class Trainer(Action):

    def __init__(self, config: CfgNode):
        self.model = build_model(config)
        if config.debug_mode:
            print(self.model)
            print(summary(self.model, (1, 1, 256, 256), device='cpu'))
        self.train_dl, self.valid_dl = self.build_loaders(config)
        self.test_dl = None
        self.should_test = False
        self.metrics = build_metrics(config)
        self.opt_t, self.opt_kws = build_optim(config)
        self.sched_t, self.sched_kws = build_sched(config, len(self.train_dl))

        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump())

        self.n_epochs = config.training.n_epochs
        self.device = torch.device(config.training.device)
        self.output_dir = config.output_dir
        self.checkpoint_every = config.training.checkpoint_every
        self.i = self.total_valid_loss = self.total_train_loss = self.total_test_loss = 0
        self.min_valid_loss = np.inf
        self.bar = self.last_checkpoint = None

        self.store = None
        self.base_exp_id = datetime.now().strftime(f'%Y%m%d_%H%M%S_MaskRCNN')

        # used by sub_classes
        self.prefix = ''
        self.as_context_manager = False

    def __enter__(self):
        self.as_context_manager = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.store is not None:
                if exc_type is KeyboardInterrupt:
                    status = 'CANCELLED'
                else:
                    status = 'ERROR'

                self.store.set_exp_status(self.exp_id, status)

            with open(f'{self.output_dir}/error.txt', 'w') as f:
                f.write(format_exception(exc_type, exc_val, exc_tb))

    @property
    def exp_id(self):
        suffix = self.prefix.rstrip('_')
        if suffix:
            suffix = '_' + suffix
        return self.base_exp_id + suffix

    def build_loaders(self, config):
        _ = self
        return build_dataloaders(config)

    @property
    def should_checkpoint(self) -> bool:
        return (self.i % self.checkpoint_every) == 0

    def prep_target(self, targets: dict):
        rv = dict()
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                v = v[0].to(self.device)
            rv[k] = v
        return rv

    def do_train(self, opt, scheduler):
        self.model.train()
        for batch in self.train_dl:
            inp = batch['image'].to(self.device)
            tgt = self.prep_target(batch['target'])
            opt.zero_grad()

            loss_dict = self.model(inp, [tgt])
            loss = sum(loss_dict.values())

            train_loss = loss.item()
            self.total_train_loss += train_loss
            loss.backward()
            opt.step()
            scheduler.step()

        self.store.add_loss_value(self.exp_id, 'train', self.i, self.total_train_loss / len(self.train_dl))

    def validate_or_test(self, dataloader, is_test: bool):

        metrics = defaultdict(list)
        valid_or_test = 'test' if is_test else 'valid'

        with torch.no_grad():
            self.model.train()
            for batch in dataloader:
                inp = batch['image'].to(self.device)
                tgt = self.prep_target(batch['target'])

                loss_dict = self.model(inp, [tgt])
                loss = sum(loss_dict.values())

                if is_test:
                    self.total_test_loss += loss.item()
                else:
                    self.total_valid_loss += loss.item()

            self.model.eval()
            for batch in dataloader:
                inp = batch['image'].to(self.device)
                tgt = self.prep_target(batch['target'])
                out = self.model(inp)
                for metric_name, metric_func in self.metrics.items():
                    for o, t in zip(out, tgt):
                        metric_value = metric_func(o, t)
                        try:
                            metric_value = to_float(metric_value)
                        except TypeError:
                            print(metric_name)
                            raise
                        metrics[f'metrics.{valid_or_test}.{metric_name}'].append(metric_value)
                        # metrics[f'metrics.by_class.{tag}.{valid_or_test}.{metric_name}'].append(metric_value)

            for key, values in metrics.items():
                self.store.add_metric_value(self.exp_id, key, self.i, np.mean(values))
        self.store.add_loss_value(
            self.exp_id, valid_or_test, self.i,
            (self.total_test_loss if is_test else self.total_valid_loss) / len(dataloader)
        )

    def do_validation(self):
        self.validate_or_test(self.valid_dl, is_test=False)

    def do_test(self):
        assert self.should_test
        assert self.test_dl is not None
        self.validate_or_test(self.test_dl, is_test=True)

    def save_dataset_contents(self, dataloader, tag):
        sources = []
        for d in dataloader:
            batch_sources = np.array(d['source'], dtype=str).flatten()
            sources.extend([os.path.normpath(s) for s in batch_sources])

        with open(f'{self.output_dir}/{self.prefix}dataset_{tag}_listing.txt', 'w') as f:
            for line in sources:
                f.write(f'{line}\n')

    def train(self):

        if not self.as_context_manager:
            print('''Trainer should ideally be used as a context manager:
```
with Trainer(cfg) as t:
    t.train()
```
so that any exceptions can be properly handled, and training status can be logged properly in the database.
''')

        self.i = self.total_valid_loss = self.total_train_loss = 0

        self.device = torch.device(self.device)
        print(f'Running on "{self.device}".')
        self.model = self.model.to(self.device)
        print(f'Saving output to "{self.output_dir}".')

        with open(f'{self.output_dir}/model.txt', 'w') as f:
            f.write(str(self.model))

        self.save_dataset_contents(self.train_dl, 'train')
        self.save_dataset_contents(self.valid_dl, 'valid')

        opt = self.opt_t(self.model.parameters(), **self.opt_kws)
        scheduler = self.sched_t(opt, **self.sched_kws)
        with Database() as self.store:
            print(f'Connected to database at "{self.store.path}".')
            self.store.set_exp_status(self.exp_id, 'TRAINING')
            self.store.set_config_file(self.exp_id, f'{self.output_dir}/config.yaml')
            self.do_validation()
            self.bar = progressbar(range(self.n_epochs), unit='epoch')
            for _i in self.bar:
                self.i = _i + 1

                self.total_train_loss = 0
                self.do_train(opt, scheduler)

                self.total_valid_loss = 0.0
                self.do_validation()

                if self.should_checkpoint:
                    self.checkpoint()
                    self.last_checkpoint = self.i

                self.update_progress()

            # always checkpoint at the end
            self.checkpoint()
            if self.should_test:
                assert self.test_dl
                # print('Loading best model state (decided based on validation set results)')
                # self.model.load_state_dict(torch.load(f'{self.output_dir}/{self.prefix}model_min_loss.pth'))
                print('Running on test dataset')
                self.do_test()

            self.store.set_exp_status(self.exp_id, 'COMPLETE')

    def act(self):
        self.train()

    def checkpoint(self):
        state_path = f'{self.output_dir}/{self.prefix}model_state_at_epoch={self.i}.pth'
        torch.save(self.model.state_dict(), state_path)
        self.store.add_state_file(self.exp_id, self.i, state_path)

    def update_progress(self):
        desc = f'{self.prefix}t:{self.total_train_loss / len(self.train_dl):.2e}|v:{self.total_valid_loss / len(self.valid_dl):.2e}|'
        if self.last_checkpoint is None:
            desc += '!*'
        else:
            desc += f'c:{self.last_checkpoint}'
            if self.i > self.last_checkpoint:
                desc += '*'
        self.bar.set_description(desc, False)
