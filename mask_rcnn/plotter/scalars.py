from matplotlib import pyplot as plt
import numpy as np

from .base import PlotterBase


class ScalarPlotter(PlotterBase):

    def __init__(self, trainer, fig_name: str, xlabels, ylabels):
        self.trainer = trainer
        self.fig_name = fig_name
        self.axes, self.fig = None, None
        self.xlabels = xlabels
        self.ylabels = ylabels
        print('labels', self.trainer.labels)
        print('tags', self.trainer.tags)

    def init(self):
        self.fig, self.axes = plt.subplots(nrows=self.trainer.model.fc.out_features)
        for ax, xl, yl in zip(self.axes, self.xlabels, self.ylabels):
            plt.sca(ax)
            plt.xlabel(xl)
            plt.ylabel(yl)

    def plot_targets(self, batch, outputs):
        plt.figure(self.fig)
        targets = batch['target'].detach().cpu().squeeze().numpy()
        outputs = outputs.detach().cpu().squeeze().numpy()

        for t, o, tag in zip(targets, outputs, batch['tag']):
            # conc = label.replace('wt%', '')
            # conc = conc.replace('Dense', '20')\
            #     .replace('LowMidConc', '15')\
            #     .replace('MidConc', '10')\
            #     .replace('Sparse', '5')
            # conc = float(conc)
            size = np.mean(np.array(tag.split()[1].split('-'), dtype=float))
            colour = plt.get_cmap('viridis')(size / 1e3)
            for j, (tj, oj) in enumerate(zip(t, o)):
                plt.sca(self.axes[j])
                plt.plot(tj, oj, 'o', color=colour)

    def finalise(self):
        plt.figure(self.fig)

        for ax in self.axes:
            plt.sca(ax)
            plt.axline((0, 0), slope=1, color='k', ls='--')

        plt.tight_layout()
        plt.savefig(f'{self.trainer.output_dir}/{self.trainer.prefix}{self.fig_name}_at_epoch={self.trainer.i}.pdf')
        plt.close()
