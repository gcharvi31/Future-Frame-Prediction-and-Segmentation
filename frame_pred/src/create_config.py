import os
import json

def create_config_json(cfg, output_file):
    with open(output_file, "w") as outfile:
        json.dump(cfg, outfile)

cfg = {
    'meta':
    {
        'board_path': 'board',
        'board_path_epoch': 'board_epoch',
        'data_path_local_train': '../data/ffp_sample_train',
        'data_path_local_test': '../data/ffp_sample_test',
        'data_path_hpc_train': '../../../../../vast/cg4177/raw_data1/Dataset_Student/unlabeled/test',
        'data_path_hpc_test': '../../../../../vast/cg4177/raw_data1/Dataset_Student/val',
    },
    'train':
    {
        'epochs': 2,
        'batch_size': 2,
        'lr': 0.001,
        'print_frequency': 1
    },
    'model':
    {
        'input_size': [40, 60],
        'input_dim': 3,
        'hidden_dim': 64,
        'kernel_size': [3, 3],
        'future_frames': 11
    }
}

create_config_json(cfg=cfg, output_file='config_local.json')