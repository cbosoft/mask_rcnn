import os.path

from matplotlib import pyplot as plt

from .base import PlotterBase


class VectorPlotter(PlotterBase):

    def __init__(self, trainer,
                 fig_name: str,
                 xkey: str, ykey: str,
                 xlabel: str, ylabel: str,
                 xscale='linear', yscale='linear', f_outputs=None, f_targets=None,
                 n_tags=-1):
        self.axes, self.fig = None, None
        self.trainer = trainer
        self.fig_name = fig_name

        self.xkey = xkey
        self.ykey = ykey

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale

        self.f_outputs = f_outputs
        self.f_targets = f_targets

        self.n_tags = n_tags
        self.tags = set()

    def init(self):
        if self.n_tags <= 0:
            self.fig = plt.figure()
            self.axes = [plt.gca()]
        else:
            r = c = int(self.n_tags**0.5)
            while r*c < self.n_tags:
                r += 1
            self.fig, self.axes = plt.subplots(nrows=r, ncols=c, figsize=(r*4, c*4))
            self.axes = self.axes.flatten()

    def plot_targets_unkn_tags(self, batch, outputs):
        plt.figure(self.fig)
        x_values = batch[self.xkey].detach().cpu().squeeze().numpy()
        y_values_target = batch[self.ykey].detach().cpu().squeeze().numpy()
        y_values_output = outputs.detach().cpu().squeeze().numpy()
        for x, yt, yo in zip(x_values, y_values_target, y_values_output):
            if self.f_outputs is not None:
                yo = self.f_outputs(x, yo)
            if self.f_targets is not None:
                yt = self.f_targets(x, yt)
            plt.plot(x, yt, 'o', color='k')
            plt.plot(x, yo, color='b')

        for tag in batch['tag']:
            self.tags.add(tag)

    def plot_targets_known_tags(self, batch, outputs):
        if isinstance(self.tags, set):
            self.tags = list(self.tags)

        x_values = batch[self.xkey].detach().cpu().squeeze().numpy()
        y_values_target = batch[self.ykey].detach().cpu().squeeze().numpy()
        y_values_output = outputs.detach().cpu().squeeze().numpy()

        for tag, x, yt, yo in zip(
            batch['tag'],
            x_values,
            y_values_target,
            y_values_output,
        ):
            i = self.tags.index(tag)
            plt.sca(self.axes[i])
            if self.f_outputs is not None:
                yo = self.f_outputs(x, yo)
            if self.f_targets is not None:
                yt = self.f_targets(x, yt)
            plt.title(tag)
            plt.plot(x, yt, 'o', color='k')
            plt.plot(x, yo, color='b', alpha=0.5)

    def plot_targets(self, batch, outputs):
        if self.n_tags < 0:
            self.plot_targets_unkn_tags(batch, outputs)
        else:
            self.plot_targets_known_tags(batch, outputs)

    def finalise(self):
        if self.tags:
            self.n_tags = len(self.tags)
            self.tags = sorted(self.tags)

        plt.figure(self.fig)
        plt.suptitle(os.path.basename(self.trainer.output_dir))
        for ax in self.axes:
            plt.sca(ax)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.xscale(self.xscale)
            plt.yscale(self.yscale)
        plt.tight_layout()
        plt.savefig(f'{self.trainer.output_dir}/{self.trainer.prefix}{self.fig_name}_at_epoch={self.trainer.i}.pdf')
        plt.close()
