import argparse
import numpy as np
import random
import torch
import yaml
from munch import Munch
from tqdm import tqdm

import util.utils as utils
from evaluation import ScanNetEval
from model.softgroup import SoftGroup


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    args = parser.parse_args()
    return args


def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy


def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    pos_inds = seg_gt_all != -100
    seg_gt_all = seg_gt_all[pos_inds]
    seg_pred_all = seg_pred_all[pos_inds]
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) & (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)
    iou_tensor = torch.tensor(iou_list)
    miou = iou_tensor.mean()
    return miou


if __name__ == '__main__':
    test_seed = 567
    random.seed(test_seed)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed_all(test_seed)

    args = get_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, 'r')))
    torch.backends.cudnn.enabled = False

    model = SoftGroup(**cfg.model)
    print(f'Load state dict from {args.checkpoint}')
    model = utils.load_checkpoint(model, args.checkpoint)
    model.cuda()

    from data.scannetv2_inst import Dataset
    dataset = Dataset(**cfg.data.test)
    dataset.valLoader()
    dataloader = dataset.val_data_loader
    all_preds, all_gts = [], []
    with torch.no_grad():
        model = model.eval()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            ret = model(batch)
            all_preds.append(ret['det_ins'])
            all_gts.append(ret['gt_ins'])
        scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(all_preds, all_gts)
