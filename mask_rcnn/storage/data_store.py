from typing import Dict
import json
from traceback import format_exception
import re

import numpy as np
from matplotlib import pyplot as plt


class DataStore:

    def __init__(self, output_dir: str, also_plot=True, prefix=''):
        self._data: Dict[str, dict] = {}
        self.output_dir = output_dir
        self.also_plot = also_plot
        self.prefix = prefix
        self._metadata_res = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save()
        if self.also_plot:
            self.plot()
        if any([a is not None for a in args]):
            with open(f'{self.output_dir}/error_log.txt', 'w') as f:
                f.write(f'Run ended with error: \n{format_exception(*args)}')

    def save(self):
        fn = f'{self.output_dir}/{self.prefix}raw_data.json'
        with open(fn, 'w') as f:
            json.dump(self._data, f, indent=2)

    def _plot_timeseries(self):
        single_values = dict()
        for name, data in self._data.items():
            data: dict

            x = data['x']
            y = data['y']

            ylabel = data.get('ylabel', name)
            xlabel = data.get('xlabel', 'batch or epoch')

            if len(x) == 1:
                single_values[ylabel] = y
                continue

            if not len(x):
                continue

            single_values[ylabel] = y[-1]

            if not data.get('plot_timeseries', True):
                continue

            plt.figure()
            plt.plot(x, y)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if data.get('logx', False):
                plt.xscale('log')
            if data.get('logy', False):
                plt.yscale('log')
            else:
                plt.ylim(bottom=(0.0 if min(y) >= 0.0 else -0.1))
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/fig_{self.prefix}{name}.pdf')
            plt.close()

        with open(f'{self.output_dir}/metrics.txt', 'w') as f:
            for name, value in single_values.items():
                f.write(f'{name} = {value}\n')

    def _plot_final_subset(self, name, plotname):
        metric_values = [d['y'][-1] for n, d in self._data.items() if n.startswith(name)]
        metric_names = [n.replace(name, '') for n in self._data if n.startswith(name)]
        if not metric_names:
            return
        x = np.arange(len(metric_names))
        plt.figure()
        plt.bar(x, metric_values)
        plt.xticks(x, metric_names, rotation=90)
        plt.ylim(bottom=(0.0 if min(metric_values) >= 0.0 else -0.1))
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig_{self.prefix}{plotname}.pdf')
        plt.close()

    def _plot_final_metrics(self):
        self._plot_final_subset('metrics.regressor.', 'final_metrics_regression')
        self._plot_final_subset('metrics.classifier.', 'final_metrics_classification')

    def _plot_losses(self, timeframe):
        timeframe_lbls = dict(
            per_epoch='Epochs',
            per_batch='Batches'
        )
        assert timeframe in timeframe_lbls
        train_k = f'loss.{timeframe}.train'
        valid_k = f'loss.{timeframe}.valid'
        if train_k not in self._data or valid_k not in self._data:
            return

        train_loss = self._data[train_k]
        valid_loss = self._data[valid_k]

        plt.figure()
        plt.plot(
            train_loss['x'],
            train_loss['y'],
            label='Training'
        )
        plt.plot(
            valid_loss['x'],
            valid_loss['y'],
            label='Valid'
        )
        plt.legend()
        plt.xlabel(f'{timeframe_lbls[timeframe]} [#]')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fig_{self.prefix}loss.{timeframe}.combined.pdf')
        plt.close()

    def plot(self):
        self._plot_timeseries()
        self._plot_final_metrics()
        self._plot_losses('per_epoch')

    def get_new_data_point(self, name) -> dict:
        rv = dict()
        for regex, md in self._metadata_res:
            if regex.match(name):
                rv = {**rv, **md}
        rv['x'] = []
        rv['y'] = []
        return rv

    def add_scalar(self, name: str, x, y):
        if name not in self._data:
            self._data[name] = self.get_new_data_point(name)

        self._data[name]['x'].append(x)
        self._data[name]['y'].append(y)

    def add_metadata(self, name: str, **metadata):
        self._data[name] = dict(
            **self._data.get(name, self.get_new_data_point(name)),
            **metadata
        )

    def add_metadata_re(self, regex, **metadata):
        self._metadata_res.append((re.compile(regex), metadata))

    def get_data(self):
        return self._data
