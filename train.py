import os
from sys import setswitchinterval
import time
import argparse
from datetime import datetime


from PIL import Image
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.coco import COCO, COCO_eval
from datasets.pascal import PascalVOC, PascalVOC_eval
from datasets.ubpmc.ubpmc import UBPMCDataset_Bar, UBPMCDataset_Line

# from nets.resdcn import get_pose_net
from nets.hourglass import get_hourglass

from utils.losses import _neg_loss, _ae_loss, _reg_loss, Loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature

from lib.nms.nms import soft_nms, soft_nms_merge

from chart_models.bar.bar_chart_kp_detection import get_bar_chart_model,bar_chart_loss

from chart_models.bar.Bar_Rule import GroupBarRaw

# Training settings
parser = argparse.ArgumentParser(description='cornernet')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--dataset', type=str, default='ubpmc', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='ubpmc_bar')

parser.add_argument('--img_size', type=int, default=511)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--lr_step', type=str, default='45,60')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=70)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)


parser.add_argument('--chart_type', type=str, default='bar')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
  saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

  num_gpus = torch.cuda.device_count()
  if cfg.dist:
    cfg.device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    cfg.device = torch.device('cuda')

  print('Setting up data...')
  Dataset_Dict = {
    'ubpmc_bar':UBPMCDataset_Bar,
    'ubpmc_line':UBPMCDataset_Line,
    'coco':COCO,
    'pascal':PascalVOC
  }
  if cfg.arch in Dataset_Dict:
    Dataset = Dataset_Dict[cfg.arch]
  # Dataset = COCO if cfg.dataset == 'coco' else UBPMCDataset_Bar#PascalVOC
  #train_dataset = Dataset(cfg.data_dir, 'val', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
  train_dataset = Dataset(cfg.data_dir,is_Training=True)

  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus
                                             if cfg.dist else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             drop_last=True,
                                             sampler=train_sampler if cfg.dist else None)
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
  val_dataset = Dataset_eval(cfg.data_dir,is_Training=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=False, num_workers=1, pin_memory=True,
                                           )#collate_fn=val_dataset.collate_fn

  print('Creating model...')
  if 'bar' in cfg.arch:
    model = get_bar_chart_model('tiny_hourglass')


  # if cfg.dist:
  #   # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  #   model = model.to(cfg.device)
  #   model = nn.parallel.DistributedDataParallel(model,
  #                                               device_ids=[cfg.local_rank, ],
  #                                               output_device=cfg.local_rank)
  # else:
  #   # todo don't use this, or wrapped it with utils.losses.Loss() !
  #   model = nn.DataParallel(model).to(cfg.device)
  model = model.to(cfg.device)
  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

  def train(epoch):
    print('\n%s Epoch: %d' % (datetime.now(), epoch))
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
      for k in batch:
        batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

      outputs = model(batch)#batch['image'])
      if 'bar' in cfg.arch:
        loss = bar_chart_loss(outputs,batch)

      optimizer.zero_grad()
      loss[0].backward()
      optimizer.step()
      (focal_loss,reg_loss,pull_loss,push_loss) = loss[1]
      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - tic
        tic = time.perf_counter()
        print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
              ' focal_loss= %.5f pull_loss= %.5f push_loss= %.5f reg_loss= %.5f' %
              (focal_loss.item(), pull_loss.item(), push_loss.item(), reg_loss.item()) +
              ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

        step = len(train_loader) * epoch + batch_idx
        summary_writer.add_scalar('focal_loss', focal_loss.item(), step)
        summary_writer.add_scalar('pull_loss', pull_loss.item(), step)
        summary_writer.add_scalar('push_loss', push_loss.item(), step)
        summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
    return

  def val_map(epoch):
    print('\n%s Val@Epoch: %d' % (datetime.now(), epoch))
    model.eval()
    # torch.cuda.empty_cache()

    results = {}
    with torch.no_grad():
      #TODO LOAD batch['image'] from PIL image
      for batch_idx, batch in enumerate(train_loader):
        for k in batch:
          batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

          outputs = model(batch)
          tl_detections = outputs[0]
          br_detection = outputs[1]

          bar_data = GroupBarRaw(batch['image'],tl_detections,br_detection)
          #TODO WRITE BBOXES TO FILE FOR EVALUATION

  print('Starting training...')
  for epoch in range(1, cfg.num_epochs + 1):
    train_sampler.set_epoch(epoch)
    train(epoch)
    
    ################## commented for now, output will be generated later in test model script
    if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
      val_map(epoch)
    print(saver.save(model.module.state_dict(), 'checkpoint'))
    lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

  summary_writer.close()


if __name__ == '__main__':
  with DisablePrint(local_rank=cfg.local_rank):
    main()
