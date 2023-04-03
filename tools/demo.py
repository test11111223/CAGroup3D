import argparse
import datetime
import os
import re
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from MinkowskiEngineBackend._C import is_cuda_available

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--demo_tag', type=str, default='default', help='eval tag for this experiment')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    #cfg.LOCAL_RANK = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0

    # np.random.seed(1024) # NOTE: set seed 
    common_utils.set_random_seed(0)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def demo_single_ckpt(demo_dataset, model, epoch_id, args, logger):
    # load checkpoint
    to_cpu = False if is_cuda_available() else True

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=to_cpu)
    # NOTE(lihe): debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model.refresh_weights() # remember to remove this line!
    if is_cuda_available():
        model.cuda()
    else:
        model.cpu()

    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])

            #dict_keys(['frame_id', 'gt_boxes', 'points', 'instance_mask', 'semantic_mask', 'axis_align_matrix', 'use_lead_xyz', 'batch_size'])
            #raise AssertionError(data_dict.keys())
          
            # NOTE: also feed cur_epoch_id
            data_dict['cur_epoch'] = epoch_id

            pred_dicts, _ = model.forward(data_dict)

            try:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)
            except:
                continue

    logger.info('Demo done.')

def main():
    args, cfg = parse_config()
    dist_demo = False
    args_batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_output_dir = output_dir / 'demo'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    demo_output_dir = demo_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']

    if args.demo_tag is not None:
        demo_output_dir = demo_output_dir / args.demo_tag

    demo_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = demo_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # GPU count or CPU count. However CPU count is capped at 24 (OMP_NUM_THREADS).
    args_workers = max(os.cpu_count(), 24)

    # Some parameters are hardcoded because it is not used.
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args_batch_size,
        dist=dist_demo, workers=args_workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        demo_single_ckpt(test_set, model, epoch_id, args, logger)


if __name__ == '__main__':
    main()
