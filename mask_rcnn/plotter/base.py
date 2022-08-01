class PlotterBase:

    def init(self):
        raise NotImplementedError

    def plot_targets(self, batch, outputs):
        raise NotImplementedError

    def finalise(self):
        raise NotImplementedError
