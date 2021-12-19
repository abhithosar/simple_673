import os
import argparse
from datetime import datetime

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import pickle
import torch.nn as nn
import torch.utils.data

from datasets.coco import COCO_eval
from datasets.pascal import PascalVOC_eval

from datasets.pascal import PascalVOC, PascalVOC_eval
from datasets.ubpmc.ubpmc import UBPMCDataset_Bar, UBPMCDataset_Line
from nets.hourglass import get_hourglass
# from nets.resdcn import get_pose_net

from utils.keypoint import _decode, _rescale_dets
from utils.summary import create_logger

from lib.nms.nms import soft_nms, soft_nms_merge

from chart_models.bar.bar_chart_kp_detection import get_bar_chart_model,bar_chart_loss,get_inference_on_bar
from chart_models.line.line_chart_kp_detection import get_inference_on_line, get_line_chart_model, line_chart_loss
# Training settings
parser = argparse.ArgumentParser(description='cornernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
# parser.add_argument('--arch', type=str, default='ubpmc_bar')
parser.add_argument('--arch', type=str, default='ubpmc_line')

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')

parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--ae_threshold', type=float, default=0.5)
parser.add_argument('--nms_threshold', type=float, default=0.5)
parser.add_argument('--w_exp', type=float, default=10)

parser.add_argument('--num_workers', type=int, default=1)

parser.add_argument('--chart_type', type=str, default='line')

parser.add_argument('--train_db',type=str,default='ubpmc',choices=['synth', 'ubpmc'])
parser.add_argument('--test_db',type=str,default='ubpmc',choices=['synth', 'ubpmc'])

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, f'checkpoint_{cfg.arch}.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def main():
  logger = create_logger(save_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = False
  cfg.device = torch.device('cuda')

  print('Setting up data...')


  dataset_splits = None

  with open('scrap/db_split.pickle', 'rb') as handle:
    dataset_splits = pickle.load(handle)
  
  Dataset_Eval_Dict = {
    'ubpmc_bar':UBPMCDataset_Bar,#UBPMCDataset_Bar_Eval,
     'ubpmc_line':UBPMCDataset_Line,
    'coco':COCO_eval,
    'pascal':PascalVOC_eval
  }
  if cfg.arch in Dataset_Eval_Dict:
    Dataset_eval = Dataset_Eval_Dict[cfg.arch]

  # Dataset_eval = COCO_eval if cfg.dataset == 'coco' else UBPMCDataset_Bar#PascalVOC_eval
  #val_dataset = Dataset_eval(cfg.data_dir, 'val', test_scales=[1.], test_flip=False)
  val_dataset = Dataset_eval(cfg.data_dir,is_Training=False,dataset=dataset_splits,
                          arch=cfg.arch,
                          is_inference=True,
                          testdb='ubpmc')
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=False, num_workers=1, pin_memory=True,
                                           )#collate_fn=val_dataset.collate_fn

  print('Creating model...')
  if 'bar' in cfg.arch:
    model = get_bar_chart_model('tiny_hourglass',True)
  if 'line' in cfg.arch:
    model = get_line_chart_model('tiny_hourglass',True)



  model = model.to(cfg.device)
  model.load_state_dict(torch.load(cfg.pretrain_dir))
  print('loaded pretrained model from %s !' % cfg.pretrain_dir)

  print('validation starts at %s' % datetime.now())
  model.eval()
  results = {}
  with torch.no_grad():
    index = 0 
    length = val_dataset.num_samples
    while index < length:
      btch = val_dataset.__getitem__(index)
      index+=1
      if 'line' in cfg.arch:
        outs = get_inference_on_line(model,btch['image'])
      elif 'bar' in cfg.arch:
        outs = get_inference_on_bar(model,btch['image'])

      iss = 0

    # for batch_idx, batch_val in enumerate(val_loader):
    #   # if 'bar' in cfg.arch:
    #   #     batch_val['image'] = batch_val['image'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['hmap_tl'] = batch_val['hmap_tl'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['hmap_br'] = batch_val['hmap_br'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['regs_tl'] = batch_val['regs_tl'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['regs_br'] = batch_val['regs_br'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['inds_tl'] = batch_val['inds_tl'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['inds_br'] = batch_val['inds_br'].to(device=cfg.device, non_blocking=True)
    #   #     batch_val['ind_masks'] = batch_val['ind_masks'].to(device=cfg.device, non_blocking=True)
      
    #   out1,out2 = get_inference_on_bar(model,batch_val['image'])
    #   ing = 0
 # eval_results = dataset.run_eval(results, save_dir=cfg.ckpt_dir)
 # print(eval_results)
  #print('validation ends at %s' % datetime.now())


if __name__ == '__main__':
  main()
