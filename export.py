import sys
import os
import argparse

from mask_rcnn.config import get_config
from mask_rcnn.model import build_model

import torch
import onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('state_path', type=str)
    parser.add_argument('output_path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.state_path)
    if not args.output_path.endswith('.onnx'):
        print('adding .onnx extension')
        args.output_path = args.output_path + '.onnx'
    config_path = os.path.dirname(args.state_path) + '/config.yaml'
    assert os.path.exists(config_path)
    config = get_config()
    config.merge_from_file(config_path)
    config.model.state = args.state_path
    config.model.backbone.pretrained = False
    config.training.device = 'cpu'

    print('building model...')
    model = build_model(config, True)
    print('... done')

    print('exporting model...')
    dummy_input = torch.randn(1, 3, 640, 480)
    outp = args.output_path
    torch.onnx.export(model, dummy_input, outp)
    print('... done')
    
    print('checking model...')
    model = onnx.load(outp)
    onnx.checker.check_model(model)
    print('... done')
    print('all done!')
