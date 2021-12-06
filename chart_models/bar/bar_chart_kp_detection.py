import torch
import torch.nn as nn
from nets.hourglass import get_hourglass
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature

from utils.losses import _neg_loss, _ae_loss, _reg_loss, Loss


def get_bar_chart_model(hourglass_framework_type,):
    model = get_hourglass[hourglass_framework_type]
    return model

def bar_chart_loss(preds,target):
    batch = target
    hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*preds)

    embd_tl = [_tranpose_and_gather_feature(e, batch['inds_tl']) for e in embd_tl]
    embd_br = [_tranpose_and_gather_feature(e, batch['inds_br']) for e in embd_br]
    regs_tl = [_tranpose_and_gather_feature(r, batch['inds_tl']) for r in regs_tl]
    regs_br = [_tranpose_and_gather_feature(r, batch['inds_br']) for r in regs_br]

    focal_loss = _neg_loss(hmap_tl, batch['hmap_tl']) + \
                   _neg_loss(hmap_br, batch['hmap_br'])
    reg_loss = _reg_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
                 _reg_loss(regs_br, batch['regs_br'], batch['ind_masks'])
    pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ind_masks'])

    loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
    return (loss,(focal_loss,reg_loss,pull_loss,push_loss))
