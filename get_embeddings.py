#!/usr/bin/env python

import argparse
import os
import pickle

import numpy as np
import torch
from torchvision.models.resnet import resnet18, ResNet18_Weights
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model
from mask_rcnn.dataset.build import build_dataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default='state.pt', help='Name or path of state file. If no path separator is present, assumes is in same dir as config.')
    parser.add_argument('--layer', type=str, default='pool', help='Layer to extract from backbone.')
    parser.add_argument('config', type=str, help='Path to config file for model.')
    parser.add_argument('dataset', nargs='*', type=str, help='Path(s) to JSON/COCO datasets for evaluation.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    assert os.path.isfile(config_path), config_path
    exp_path = os.path.dirname(config_path)
    if os.sep in args.state:
        state_path = args.state
    else:
        state_path = os.path.join(exp_path, args.state)
    assert os.path.isfile(state_path), state_path

    for ds_path in args.dataset:
        assert os.path.isfile(ds_path), ds_path

    ############################################################################

    model_name = os.path.basename(os.path.dirname(config_path))
    cfg = get_config()
    cfg.merge_from_file(config_path)
    finalise(cfg)

    model = build_model(cfg)
    backbone = model.backbone

    datasets = dict()
    datasets[f'{model_name}_training_set'] = build_dataset(cfg)

    for ds_path in args.dataset:
        ds_name, _ = os.path.splitext(os.path.basename(ds_path))
        cfg.defrost()
        cfg.data.pattern = [ds_path]
        datasets[ds_name] = build_dataset(cfg)

    data = list(datasets.items())
    bar = tqdm(data, unit='datasets')
    for ds_name, ds in bar:
        bar.set_description(ds_name)
        dataloader = DataLoader(ds, shuffle=False, batch_size=1, collate_fn=collate_fn)

        embeddings = []
        for i, datapoint in enumerate(dataloader):
            fn = datapoint['source'][0]
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (512, 512))
            img = torch.tensor(img).permute(2, 0, 1).float()/255
            img = torch.unsqueeze(img, 0)
            one_embedding = backbone(img)[args.layer]
            one_embedding = one_embedding.detach().cpu().numpy().flatten()
            embeddings.append(one_embedding)
        embeddings = np.array(embeddings)

        with open(f'embeddings/embeddings_{ds_name}.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
