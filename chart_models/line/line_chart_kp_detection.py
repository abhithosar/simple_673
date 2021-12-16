import torch
import torch.nn as nn
from nets.hourglass import get_hourglass
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature

from utils.losses import _ae_line_loss,_neg_loss, _ae_loss, _reg_loss, Loss, AELossLine


def get_line_chart_model(hourglass_framework_type,for_inference):
    model = get_hourglass(hourglass_framework_type,'line',for_inference)
    return model

def line_chart_loss(preds,target):
    batch = target
    pull_weight = 1e-1
    push_weight = 1e-1
    _lambda = 4
    _lambd = 2

    stride = 5

    key_heat, hybrid_heat, key_tag, key_tag_grouped, key_regr = zip(*preds)
    
    focal_loss = 0

    focal_loss = _neg_loss(key_heat, batch['key_heatmap']) + \
                _neg_loss(hybrid_heat, batch['hybrid_heatmaps'])
    
    pull_loss = 0
    push_loss = 0

    for key_tag_grped in key_tag_grouped:
        pull,push = _ae_line_loss(key_tag_grped,batch['tag_masks_grouped'])
        pull_loss += pull
        push_loss += push
    pull_loss = pull_loss*pull_weight
    push_loss = push_loss * push_weight

    reg_loss = _reg_loss(key_regr, batch['key_regrs'], batch['tag_masks'])
    loss = (focal_loss + pull_loss + push_loss + reg_loss) / len(key_heat)    
    return loss.unsqueeze(0)
