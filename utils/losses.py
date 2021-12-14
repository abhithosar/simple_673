import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.keypoint import _tranpose_and_gather_feature



def _ae_line_loss(tag_full, mask_full):
    # mask_full [batch, Max_group, Max_len]
    # tag_full  [batch, Max_group, Max_len]
    pull = 0
    push = 0
    tag_full = torch.squeeze(tag_full)
    tag_full[1-mask_full] = 0
    num = mask_full.sum(dim=2, keepdim=True).float()
    tag_avg = tag_full.sum(dim=2, keepdim=True) / num
    pull = torch.pow(tag_full - tag_avg, 2) / (num + 1e-4)
    pull = pull[mask_full].sum()

    tag_avg = torch.squeeze(tag_avg)
    mask = mask_full.sum(dim=2)
    mask = mask.gt(1)
    num = mask.sum(dim=1, keepdim=True).float()
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)

    dist = tag_avg.unsqueeze(1) - tag_avg.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _neg_loss(preds, targets):
  pos_inds = targets == 1  # todo targets > 1-epsilon ?
  neg_inds = targets < 1  # todo targets < 1-epsilon ?

  neg_weights = torch.pow(1 - targets[neg_inds], 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)


def _ae_loss(embd0s, embd1s, mask):
  num = mask.sum(dim=1, keepdim=True).float()  # [B, 1]

  pull, push = 0, 0
  for embd0, embd1 in zip(embd0s, embd1s):
    embd0 = embd0.squeeze()  # [B, num_obj]
    embd1 = embd1.squeeze()  # [B, num_obj]

    embd_mean = (embd0 + embd1) / 2

    embd0 = torch.pow(embd0 - embd_mean, 2) / (num + 1e-4)
    embd0 = embd0[mask].sum()
    embd1 = torch.pow(embd1 - embd_mean, 2) / (num + 1e-4)
    embd1 = embd1[mask].sum()
    pull += embd0 + embd1

    push_mask = (mask[:, None, :] + mask[:, :, None]) == 2  # [B, num_obj, num_obj]
    dist = F.relu(1 - (embd_mean[:, None, :] - embd_mean[:, :, None]).abs(), inplace=True)
    dist = dist - 1 / (num[:, :, None] + 1e-4)  # substract diagonal elements
    dist = dist / ((num - 1) * num + 1e-4)[:, :, None]  # total num element is n*n-n
    push += dist[push_mask].sum()
  return pull / len(embd0s), push / len(embd0s)


def _reg_loss(regs, gt_regs, mask):
  num = mask.float().sum() + 1e-4
  mask = mask[:, :, None].expand_as(gt_regs)  # [B, num_obj, 2]
  loss = sum([F.smooth_l1_loss(r[mask], gt_regs[mask], reduction='sum') / num for r in regs])
  return loss / len(regs)

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x
    
class Loss(nn.Module):
  def __init__(self, model):
    super(Loss, self).__init__()
    self.model = model

  def forward(self, batch):
    outputs = self.model(batch['image'])
    hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*outputs)

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
    return loss.unsqueeze(0), outputs

class AELossLine(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, lamda=4, lamdb=2):
        super(AELossLine, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_line_loss
        self.regr_loss   = _reg_loss
        self.lamda = lamda
        self.lamdb = lamdb

    def forward(self, outs, targets):
        stride = 5
        key_heats = outs[0::stride]
        hybrid_heats = outs[1::stride]
        key_tags  = outs[2::stride]
        key_tags_grouped  = outs[3::stride]
        key_regrs = outs[4::stride]


        gt_key_heat = targets[0]
        gt_hybrid_heat = targets[1]
        gt_mask    = targets[2]
        gt_mask_grouped = targets[3]
        gt_key_regr = targets[4]

        # focal loss
        focal_loss = 0

        key_heats = [_sigmoid(t) for t in key_heats]
        hybrid_heats = [_sigmoid(b) for b in hybrid_heats]

        focal_loss += self.focal_loss(key_heats, gt_key_heat, self.lamda, self.lamdb)
        focal_loss += self.focal_loss(hybrid_heats, gt_hybrid_heat, self.lamda, self.lamdb)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for key_tag_grouped in key_tags_grouped:
            pull, push = self.ae_loss(key_tag_grouped, gt_mask_grouped)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for key_regr in key_regrs:
            regr_loss += self.regr_loss(key_regr, gt_key_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(key_heats)
        return loss.unsqueeze(0)