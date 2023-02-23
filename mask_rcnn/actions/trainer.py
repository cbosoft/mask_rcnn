import os
from datetime import datetime
from traceback import format_exception

import torch
import numpy as np
from torchinfo import summary
from mldb import Database

from ..coco_evaluation import coco_eval_datasets, update_coco_datasets_from_batch
from ..visualisation import visualise_valid_batch
from ..config import CfgNode, as_hyperparams
from ..progress_bar import progressbar
from ..model import build_model
from ..dataset import build_dataloaders
from ..optim import build_optim
from ..sched import build_sched
from ..augmentations import build_augmentations
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
        self.opt_t, self.opt_kws = build_optim(config)
        self.transform = build_augmentations(config)

        self.sched_t, self.sched_kws = build_sched(
            config,
            # if train_dl is None, then just put a random number,
            # these kwargs will be overwritten elsewehere
            len(self.train_dl) if self.train_dl is not None else 100
        )


        with open(f'{config.output_dir}/config.yaml', 'w') as f:
            f.write(config.dump())

        self.n_epochs = config.training.n_epochs
        self.device = torch.device(config.training.device)
        self.output_dir = config.output_dir
        self.checkpoint_every = config.training.checkpoint_every
        self.visualise_every = config.training.visualise_every
        self.should_show_visualisations = config.training.show_visualisations
        self.i = self.total_valid_loss = self.total_train_loss = self.total_test_loss = 0
        self.min_valid_loss = np.inf
        self.bar = self.last_checkpoint = None
        self.displayed_metrics = {}

        self.store = None
        self.base_exp_id = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_MaskRCNN')

        self.hyperparams = as_hyperparams(config)

        # used by sub_classes
        self.prefix = ''
        self.as_context_manager = False

    def __enter__(self):
        self.as_context_manager = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # need to re-open conn to db as it is closed before getting here
        with Database() as self.store:
            if exc_type is not None:
                if self.store is not None:
                    if exc_type is KeyboardInterrupt:
                        status = 'CANCELLED'
                    else:
                        status = 'ERROR'

                    self.store.set_exp_status(self.exp_id, status)

                with open(f'{self.output_dir}/error.txt', 'w') as f:
                    f.write(''.join(format_exception(exc_type, exc_val, exc_tb)))

            # Always checkpoint.
            try:
                self.checkpoint()
            except:
                pass  # If checkpoint already exists, the above will raise an exception. Ignore it.

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

    def prep_target(self, targets):

        if not isinstance(targets, list):
            targets = [targets]

        rvs = []
        for t in targets:
            rv = dict()
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.device)
                rv[k] = v
            rvs.append(rv)
        return rvs

    def do_train(self, opt, scheduler):
        self.model.train()
        for batch in self.train_dl:
            inp = batch['image']
            tgt = batch['target']
            inp, tgt = self.transform(inp, tgt)
            if isinstance(inp, list):
                inp = [i.to(self.device) for i in inp]
            else:
                inp = [inp.to(self.device)]
            tgt = self.prep_target(tgt)
            opt.zero_grad()

            loss_dict = self.model(inp, tgt)
            loss = sum(loss_dict.values())

            train_loss = loss.item()
            self.total_train_loss += train_loss
            loss.backward()
            opt.step()
            scheduler.step()

        self.store.add_loss_value(self.exp_id, 'train', self.i, self.total_train_loss / len(self.train_dl))

    def validate_or_test(self, dataloader, is_test: bool):

        assert dataloader is not None, 'Error: cannot run validation on None dataset'

        valid_or_test = 'test' if is_test else 'valid'

        coco_images = dict()
        coco_categories = [dict(id=i, name=f'cat{i}', supercategory=None) for i in range(1, 11)]
        coco_gt_anns = []
        coco_dt_anns = []

        with torch.no_grad():
            self.model.train()
            for batch in dataloader:
                inp = batch['image']
                if isinstance(inp, list):
                    inp = [i.to(self.device) for i in inp]
                else:
                    inp = inp.to(self.device)
                tgt = self.prep_target(batch['target'])

                loss_dict = self.model(inp, tgt)
                loss = sum(loss_dict.values())

                if is_test:
                    self.total_test_loss += loss.item()
                else:
                    self.total_valid_loss += loss.item()

            self.model.eval()
            done_vis = False
            for batch in dataloader:
                inp = batch['image']
                if isinstance(inp, list):
                    inp = [i.to(self.device) for i in inp]
                else:
                    inp = inp.to(self.device)
                tgt = self.prep_target(batch['target'])
                out = self.model(inp)

                if not done_vis and ((self.i % self.visualise_every) == 0):
                    visualise_valid_batch(
                        inp, tgt, out,
                        self.should_show_visualisations,
                        output_dir=self.output_dir,
                        epoch=self.i,
                    )
                    done_vis = True

                # Add GT, DT to coco datasets
                update_coco_datasets_from_batch(coco_images, coco_gt_anns, coco_dt_anns, tgt, out)

        coco_images = list(coco_images.values())
        for i, ann in enumerate(coco_gt_anns):
            ann['id'] = i+1
        for i, ann in enumerate(coco_dt_anns):
            ann['id'] = i+1
        coco_data_gt = dict(images=coco_images, categories=coco_categories, annotations=coco_gt_anns)
        coco_data_dt = dict(images=coco_images, categories=coco_categories, annotations=coco_dt_anns)

        coco_metrics = coco_eval_datasets(coco_data_gt, coco_data_dt)
        for k, v in coco_metrics.items():
            self.store.add_metric_value(
                self.exp_id, f'{valid_or_test}.{k}', self.i, v
            )

        self.displayed_metrics = dict(
            AP50=coco_metrics['AP50'],
            mAP=coco_metrics['mAP'],
        )

        self.store.add_loss_value(
            self.exp_id, valid_or_test, self.i,
            (self.total_test_loss if is_test else self.total_valid_loss) / len(dataloader)
        )

    def do_validation(self):
        if self.valid_dl is not None:
            self.validate_or_test(self.valid_dl, is_test=False)

    def do_test(self):
        assert self.should_test
        assert self.test_dl is not None
        self.validate_or_test(self.test_dl, is_test=True)

    def save_dataset_contents(self, dataloader, tag):
        if dataloader is None:
            return

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
        self.displayed_metrics = {}

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
            print(f'Connected to database "{self.store}"')
            self.store.set_exp_status(self.exp_id, 'TRAINING')
            self.store.set_config_file(self.exp_id, f'{self.output_dir}/config.yaml')
            self.do_validation()
            self.bar = progressbar(range(self.n_epochs), unit='epoch', ncols=80)
            for _i in self.bar:
                self.i = _i + 1

                self.total_train_loss = 0
                self.do_train(opt, scheduler)

                self.total_valid_loss = 0.0
                if self.valid_dl is not None:
                    self.do_validation()

                if self.should_checkpoint:
                    self.checkpoint()
                    self.last_checkpoint = self.i

                self.update_progress()

            if self.should_test:
                assert self.test_dl
                # print('Loading best model state (decided based on validation set results)')
                # self.model.load_state_dict(torch.load(f'{self.output_dir}/{self.prefix}model_min_loss.pth'))
                print('Running on test dataset')
                self.do_test()

            self.store.set_exp_status(self.exp_id, 'COMPLETE')
            try:
                self.checkpoint()
            except:
                pass
        self.deploy(f'{self.output_dir}/{self.prefix}model')

    def deploy(self, path_no_ext: str):
        self.model.eval()

        assert not path_no_ext.endswith('.ts')
        path = path_no_ext + '.ts'
        print(f'Deploying {self.model.__class__.__name__} (CPU) to {path!r}')
        scripted = torch.jit.script(self.model.cpu())
        torch.jit.save(scripted, path)

        if torch.cuda.is_available():
            path = path_no_ext + '_cuda.ts'
            print(f'Deploying {self.model.__class__.__name__} (CUDA) to {path!r}')
            scripted = torch.jit.script(self.model.cuda())
            scripted.save(path)

    def act(self):
        self.train()

    def checkpoint(self):
        state_path = f'{self.output_dir}/{self.prefix}model_state_at_epoch={self.i}.pth'
        torch.save(self.model.state_dict(), state_path)
        self.store.add_state_file(self.exp_id, self.i, state_path)

    def update_progress(self):
        train_loss = self.total_train_loss / len(self.train_dl)
        if self.valid_dl is not None:
            valid_loss = self.total_valid_loss / len(self.valid_dl)
            divergence = valid_loss / train_loss
            desc = f'v/t:{divergence:.2f}'
        else:
            desc = f't:{train_loss:.f}'
        for k, v in self.displayed_metrics.items():
            desc += f'|{k}:{v:.2f}'
        self.bar.set_description(desc, False)
