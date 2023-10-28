import numpy as np
import torch
import mlflow

from torch.utils.data import DataLoader, Subset

from ..dataset.build import build_dataset, collate_fn
from .trainer import Trainer, CfgNode


class CrossValidator(Trainer):

    def __init__(self, cfg: CfgNode):
        self.n_folds = cfg.xval.n_folds
        self.frac_test = cfg.data.frac_test
        self.master_dataset = None
        super().__init__(cfg)
        mlflow.set_experiment_tag('action', 'xval')

        print('Save original model state')
        torch.save(self.model.state_dict(), f'{self.output_dir}/original_state.pth')
        self.batch_size = cfg.training.batch_size

    def build_loaders(self, cfg: CfgNode):
        self.master_dataset = build_dataset(cfg)
        return None, None

    def update_schedkws_iter_count(self):
        if 'steps_per_epoch' in self.sched_kws:
            self.sched_kws['steps_per_epoch'] = len(self.train_dl)
        elif 'total_iters' in self.sched_kws:
            self.sched_kws['total_iters'] = len(self.train_dl)*self.n_epochs

    def cross_validate(self):
        n = orig_n = len(self.master_dataset)

        indices = np.arange(n)
        np.random.shuffle(indices)

        dataloader_kws = dict(batch_size=self.batch_size, collate_fn=collate_fn, drop_last=True)
        if self.frac_test > 0.0:
            test_pivot = int(n*self.frac_test)
            test_indices = indices[:test_pivot]
            indices = indices[test_pivot:]
            n = len(indices)
            test_set = Subset(self.master_dataset, test_indices)
            try:
                self.test_dl = DataLoader(test_set, **dataloader_kws)
                self.should_test = True
            except ValueError:
                print(f'Don\'t have enough data to form test set! ({orig_n})')
                raise
            self.save_dataset_contents(self.test_dl, 'test')

        if n % self.n_folds:
            n = n - (n % self.n_folds)
            indices = indices[:n]
        
        fold_epochs = np.full(self.n_folds, self.n_epochs)
        folds = np.split(indices, self.n_folds)
        for i in range(self.n_folds):
            train_indices = np.array(folds[1:]).flatten()
            valid_indices = folds[0]
            folds = np.roll(folds, 1)

            train_set = Subset(self.master_dataset, train_indices)
            valid_set = Subset(self.master_dataset, valid_indices)

            self.train_dl = DataLoader(train_set, shuffle=True, **dataloader_kws)
            self.valid_dl = DataLoader(valid_set, shuffle=False, **dataloader_kws)

            print('Reset to original model state')
            self.model.load_state_dict(torch.load(f'{self.output_dir}/original_state.pth'))

            self.prefix = f'fold{i+1}_'
            self.update_schedkws_iter_count()
            self.train()
            fold_epochs[i] = self.i

        # final training
        self.train_dl = DataLoader(self.master_dataset, **dataloader_kws)
        self.valid_dl = None

        self.test_dl = None
        self.should_test = False

        self.prefix = 'final_'
        self.update_schedkws_iter_count()
        max_fold_epochs = int(np.max(fold_epochs))
        if max_fold_epochs < self.n_epochs:
            self.n_epochs = int(max_fold_epochs*1.1)
            print(f'Folds stopped early ({max_fold_epochs}): stopping final training early too ({self.n_epochs})')
        self.train()

    def act(self):
        self.cross_validate()
