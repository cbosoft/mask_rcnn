from ..config import CfgNode
from .base import PlotterBase
from .vectors import VectorPlotter
from .scalars import ScalarPlotter

from ..util import gaussian_pdf


class MultiPlotter(PlotterBase):

    def __init__(self, *plotters: PlotterBase):
        self.plotters = plotters

    def init(self):
        for p in self.plotters:
            p.init()

    def plot_targets(self, batch, outputs):
        for p in self.plotters:
            p.plot_targets(batch, outputs)

    def finalise(self):
        for p in self.plotters:
            p.finalise()


def build_plotter(trainer) -> PlotterBase:
    return VectorPlotter(
            trainer,
            'output_v_target',
            'bins', 'targets',
            xlabel='Particle Size [$\\rm\\mu m$]',
            ylabel='Number Density [$\\rm\\mu m^{-1}$]',
            xscale='log',
    )
