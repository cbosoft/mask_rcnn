import os
import json
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from ..progress_bar import progressbar
from ..config import CfgNode, as_hyperparams
from ..model import build_model
from ..optim import build_optim
from ..sched import build_sched
from ..augmentations import build_augmentations
from ..dataset import build_dataset
from .action_base import Action


class ContrastiveTrainer(Action):

    def __init__(self, cfg: CfgNode, prefix=''):
        self.prefix = prefix
        self.base_exp_id = datetime.now().strftime(f'Mask R-CNN on %Y-%m-%d at %H-%M-%S')
        self.model = build_model(cfg)
        self.enc_pool = nn.AdaptiveAvgPool1d(128)
        if cfg.debug_mode:
            print(self.model)
            print(summary(self.model, (1, 1, 256, 256), device='cpu'))
        self.dataset = build_dataset(cfg)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, drop_last=True)
        self.opt_t, self.opt_kws = build_optim(cfg)
        self.transform = build_augmentations(cfg)

        self.sched_t, self.sched_kws = build_sched(cfg, len(self.dataloader))

        with open(f'{cfg.output_dir}/config.yaml', 'w') as f:
            f.write(cfg.dump())

        self.n_epochs = cfg.training.n_epochs
        self.device = torch.device(cfg.training.device)
        self.output_dir = cfg.output_dir
        self.checkpoint_every = cfg.training.checkpoint_every
        self.visualise_every = cfg.training.visualise_every
        self.should_show_visualisations = cfg.training.show_visualisations
        self.i = self.total_loss = 0
        self.min_valid_loss = np.inf
        self.bar = self.last_checkpoint = None
        self.displayed_metrics = {}
        self.this_epoch_metrics = {}

        self.hyperparams = as_hyperparams(cfg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always checkpoint.
        try:
            self.checkpoint()
        except:
            pass  # If checkpoint already exists, the above will raise an exception. Ignore it.

    @property
    def affix(self) -> str:
        return self.prefix.rstrip('_')

    @property
    def exp_id(self):
        suffix = self.affix
        if suffix:
            suffix = '_' + suffix
        return self.base_exp_id + suffix

    @property
    def should_checkpoint(self) -> bool:
        return (self.i % self.checkpoint_every) == 0

    @property
    def should_visualise(self) -> bool:
        return True  # self.should_show_visualisations and ((self.i % self.visualise_every) == 0)
    
    def do_train(self, loss_func, opt, scheduler):
        self.model.train()
        for batch in self.dataloader:
            a_img, b_img = batch['image']
            a_cls, b_cls = batch['cls']
    
            # TODO: transforms
            # a_prime, b_prime
            opt.zero_grad()

            a_encoding = self.enc_pool(self.model.backbone(a_img)['pool'].flatten()[None, ...])
            b_encoding = self.enc_pool(self.model.backbone(b_img)['pool'].flatten()[None, ...])
            
            loss = loss_func(a_encoding, b_encoding)
            if a_cls != b_cls:
                loss = 1e3 - loss
                loss = torch.where(loss > 0.0, loss, 0.0)
                
            self.total_loss += loss.item()
            loss.backward()
            opt.step()
            scheduler.step()

        try:
            lr = scheduler.get_last_lr()[0]
        except AttributeError:
            try:
                lr = opt.defaults['lr']
            except:
                lr = float('nan')

        loss = self.total_loss / len(self.dataloader)
        self.this_epoch_metrics['loss'] = loss
        self.this_epoch_metrics['lr'] = lr

    def do_visualise(self):
        self.model.train()
        n = len(self.dataset)
        e_len = len(self.enc_pool(torch.zeros(1, 1000)).flatten())
        data = np.zeros((n, e_len))
        classes = np.zeros(n)
        for i, item in enumerate(self.dataset):
            img = item['image']
            cls = item['cls']
            encoding = self.enc_pool(self.model.backbone(img)['pool'].flatten()[None, ...])
            encoding = encoding.flatten().detach().cpu().numpy()
            data[i, :] = encoding
            classes[i] = cls

        tsne = TSNE()
        x, y = tsne.fit_transform(data).transpose()

        plt.figure(layout='tight')
        plt.title(f'Epoch={self.i}')
        plt.scatter(x, y, c=classes)
        cb = plt.colorbar()

        ps, lbls = [], []
        for lbl, p in self.dataset.idx_by_class.items():
            ps.append(p)
            lbls.append(lbl)
        cb.set_ticks(ps, labels=lbls)
        plt.xlabel('TSNE coord 1')
        plt.ylabel('TSNE coord 2')
        plt.savefig(f'{self.output_dir}/fig_tsne_{self.i}.pdf')
        plt.savefig(f'{self.output_dir}/fig_tsne_latest.pdf')
        plt.close()


    def try_continue(self, model: torch.nn.Module, opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LRScheduler):
        checkpoint_path = f'{self.output_dir}/checkpoint.pth'
        if os.path.exists(checkpoint_path):
            model_state, epochs, optimiser_state, scheduler_state, time = torch.load(checkpoint_path)
            model.load_state_dict(model_state)
            self.i = epochs
            model.load_state_dict(model_state)
            opt.load_state_dict(optimiser_state)
            sched.load_state_dict(scheduler_state)
            print(f'Loaded previous checkpoint at epoch {epochs} from {time}')

    def train(self):
        self.i = self.total_loss = 0
        self.displayed_metrics = {}

        self.device = torch.device(self.device)
        print(f'Running on "{self.device}".')
        self.model = self.model.to(self.device)
        print(f'Saving output to "{self.output_dir}".')

        with open(f'{self.output_dir}/model.txt', 'w') as f:
            f.write(str(self.model))

        loss_func = nn.MSELoss()
        opt = self.opt_t(self.model.parameters(), **self.opt_kws)
        scheduler = self.sched_t(opt, **self.sched_kws)
        self.try_continue(self.model, opt, scheduler)

        self.bar = progressbar(
            range(self.i, self.n_epochs),
            initial=self.i, total=self.n_epochs,
            unit='epoch', ncols=80)
        
        self.do_visualise()
        for _i in self.bar:
            self.i = _i + 1
            self.this_epoch_metrics = dict(epoch=self.i)

            self.total_loss = 0
            self.do_train(loss_func, opt, scheduler)
            with open(f'{self.output_dir}/metrics.json_list', 'a') as f:
                f.write(json.dumps(self.this_epoch_metrics)+'\n')

            if self.should_checkpoint:
                self.checkpoint(opt, scheduler)
                self.last_checkpoint = self.i

            if self.should_visualise:
                self.do_visualise()

            self.update_progress()

        try:
            self.checkpoint()
        except:
            pass

    def act(self):
        self.train()

    def checkpoint(self, opt: torch.optim.Optimizer = None, sched: torch.optim.lr_scheduler.LRScheduler = None):
        state_path = f'{self.output_dir}/{self.prefix}model_state_at_epoch={self.i}.pth'
        model_state = self.model.state_dict()
        torch.save(self.model.state_dict(), state_path)
        # mlflow.log_artifact(state_path)
        if opt is not None:
            assert sched
            opt_state = opt.state_dict()
            sched_state = sched.state_dict()
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            torch.save((model_state, self.i, opt_state, sched_state, time), f'{self.output_dir}/checkpoint.pth')

    def update_progress(self):
        train_loss = self.total_loss / len(self.dataloader)
        desc = f't:{train_loss:.2f}'
        self.bar.set_description(desc, False)
