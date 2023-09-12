import argparse
import os
import pickle

from torch.utils.data import DataLoader

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model
from mask_rcnn.dataset.build import build_dataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file for model.')
    parser.add_argument('state', type=str, default='state.pt', help='Name or path of state file. If no path separator is present, assumes is in same dir as config.')
    parser.add_argument('dataset', nargs='+', type=str, help='Path(s) to JSON/COCO datasets for evaluation.')
    parser.add_argument('layer', type=str, default='pool', help='Layer to extract from backbone.')
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

    for ds in args.dataset:
        assert os.path.isfile(ds_path), ds_path

    ############################################################################

    cfg = get_config()
    cfg.merge_from_file(config_path)
    cfg.model.state = state_path
    finalise(cfg)

    model = build_model(cfg)
    backbone = model.backbone

    datasets = dict()
    datasets['training'] = build_dataset(cfg)

    for ds_path in args.dataset:
        ds_name, _ = os.path.splitext(os.path.basename(ds_path))
        cfg.defrost()
        cfg.data.pattern = [ds_path]
        datasets[ds_name] = build_dataset(cfg)

    for ds_name, ds in datasets.items():
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)

        embeddings = dict()
        for datapoint in dataloader:
            fn = datapoint['source'][0]
            one_embedding = backbone(datapoint['image'][0])
            embeddings[fn] = one_embedding[args.layer].numpy().flatten()

        with open('embeddings/embeddings_{ds_name}.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
