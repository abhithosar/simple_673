import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpool import TopPool, BottomPool, LeftPool, RightPool
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature

class pool(nn.Module):
  def __init__(self, dim, pool1, pool2):
    super(pool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)

    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)

    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)

    self.conv2 = convolution(3, dim, dim)

    self.pool1 = pool1()
    self.pool2 = pool2()

  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))

    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
    bn1 = self.bn1(self.conv1(x))

    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

class pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()

    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv = self.conv(x.float())
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu


class residual(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()

    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)

    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                              nn.BatchNorm2d(out_dim)) \
      if stride != 1 or inp_dim != out_dim else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)


# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()

    self.n = n

    curr_modules = modules[0]
    next_modules = modules[1]

    curr_dim = dims[0]
    next_dim = dims[1]

    # 上支路：重复curr_mod次residual，curr_dim -> curr_dim -> ... -> curr_dim
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
    # 分辨率本来应该在这里减半...
    self.down = nn.Sequential()
    # 重复curr_mod次residual，curr_dim -> next_dim -> ... -> next_dim
    # 实际上分辨率是在这里的第一个卷积层层降的
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
    # hourglass中间还是一个hourglass
    # 直到递归结束，重复next_mod次residual，next_dim -> next_dim -> ... -> next_dim
    if self.n > 1:
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
    # 重复curr_mod次residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
    # 分辨率在这里X2
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x):
    up1 = self.top(x)  # 上支路residual
    down = self.down(x)  # 下支路downsample(并没有)
    low1 = self.low1(down)  # 下支路residual
    low2 = self.low2(low1)  # 下支路hourglass
    low3 = self.low3(low2)  # 下支路residual
    up2 = self.up(low3)  # 下支路upsample
    return up1 + up2  # 合并上下支路

class line_cls(nn.Module):
    def __init__(self, inp_dim, cat_num):
        super(line_cls, self).__init__()
        self.mid_ = torch.nn.Linear(inp_dim, 256)
        self.final_ = fully_connected(256, cat_num)
    def forward(self, x):
        fea_dim = x.size(2)
        x = x.view(-1, fea_dim)
        mid = torch.tanh(self.mid_(x))
        final = self.final_(mid)
        return final


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class exkp(nn.Module):
  def __init__(self, n, nstack, dims, modules, num_classes=1, cnv_dim=256,c_type='bar',for_inference=False):
    super(exkp, self).__init__()
    self.chart_type = c_type
    self.for_inference = for_inference
    self.nstack = nstack

    curr_dim = dims[0]

    self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))

    self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

    self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])
  
    if self.chart_type == 'bar' or self.chart_type == 'pie':
      self.cnvs_tl = nn.ModuleList([pool(cnv_dim, TopPool, LeftPool) for _ in range(nstack)])
      self.cnvs_br = nn.ModuleList([pool(cnv_dim, BottomPool, RightPool) for _ in range(nstack)])

      # heatmap layers
      self.hmap_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
      self.hmap_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
      
      for hmap_tl, hmap_br in zip(self.hmap_tl, self.hmap_br):
        hmap_tl[-1].bias.data.fill_(-2.19)
        hmap_br[-1].bias.data.fill_(-2.19) 

      # regression layers
      self.regs_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
      self.regs_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

    elif self.chart_type == 'line':
      self.key_cnvs = nn.ModuleList([pool_cross(cnv_dim, TopPool, LeftPool, BottomPool, RightPool) for _ in range(nstack)])
      self.hybrid_cnvs = nn.ModuleList([pool_cross(cnv_dim, TopPool, LeftPool, BottomPool, RightPool) for _ in range(nstack)])
    
      self.key_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
      self.hybrid_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim  , num_classes) for _ in range(nstack)])

      for key_heat, hybrid_heat in zip(self.key_heats, self.hybrid_heats):
        key_heat[-1].bias.data.fill_(-2.19)
        hybrid_heat[-1].bias.data.fill_(-2.19)
      
      self.key_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
      self.key_tags = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])

    elif self.chart_type == "line_query":
      self.cls = nn.ModuleList([line_cls(cnv_dim*8, 2) for _ in range(nstack)])

    self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

    self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                nn.BatchNorm2d(curr_dim))
                                  for _ in range(nstack - 1)])

    self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])

    

    # embedding layers
    #self.embd_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
    #self.embd_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])

    

    

    self.relu = nn.ReLU(inplace=True)

  def forward(self, inputs):
    if self.training:
      return self.train_loop(inputs)
    else:
      if self.chart_type == 'bar':
        if self.for_inference:
          return self.bar_chart_val(inputs)
        else:
          return self.train_loop(inputs)
      
      if self.chart_type == 'line':
        if self.for_inference:
          return self.bar_chart_val(inputs)
        else:
          return self.train_loop(inputs)
      return self.test_looop()

  def train_loop(self, inputs, for_val=False):
    inter = self.pre(inputs['image'])

    outs = []
    for ind in range(self.nstack):
      kp = self.kps[ind](inter)
      cnv = self.cnvs[ind](kp)

      if self.training or ind == self.nstack - 1:
        
        if self.chart_type == 'bar': 
          cnv_tl = self.cnvs_tl[ind](cnv)
          cnv_br = self.cnvs_br[ind](cnv)

          hmap_tl, hmap_br = self.hmap_tl[ind](cnv_tl), self.hmap_br[ind](cnv_br)
          regs_tl, regs_br = self.regs_tl[ind](cnv_tl), self.regs_br[ind](cnv_br)

          #embd_tl, embd_br = self.embd_tl[ind](cnv_tl), self.embd_br[ind](cnv_br)
          

      
          #embd_tl = [_tranpose_and_gather_feature(e, inputs['inds_tl']) for e in embd_tl]
          #embd_br = [_tranpose_and_gather_feature(e, inputs['inds_br']) for e in embd_br]

          # regs_tl = [_tranpose_and_gather_feature(r, inputs['inds_tl']) for r in regs_tl]
          # regs_br = [_tranpose_and_gather_feature(r, inputs['inds_br']) for r in regs_br]
          if not for_val:
            regs_tl = _tranpose_and_gather_feature(regs_tl,inputs['inds_tl'])
            regs_br = _tranpose_and_gather_feature(regs_br,inputs['inds_br'])
          #outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])
          outs.append([hmap_tl, hmap_br, regs_tl, regs_br])

        if self.chart_type == 'pie':
          center_cnv = self.cnvs_tl[ind](cnv)
          key_cnv = self.cnvs_br[ind](cnv)

          center_heat, key_heat = self.hmap_tl[ind](center_cnv), self.hmap_br[int](key_cnv)
          center_regr, key_regr = self.regs_tl[ind](center_cnv), self.regs_br[ind](key_cnv)
          center_regr = [_tranpose_and_gather_feature(e, inputs['center_inds']) for e in center_regr]
          key_regr_tl = [_tranpose_and_gather_feature(e, inputs['key_inds_tl']) for e in key_regr]
          key_regr_br = [_tranpose_and_gather_feature(e, inputs['key_inds_br']) for e in key_regr]

          outs.append([center_heat, key_heat, center_regr, key_regr_tl, key_regr_br])
        
        if self.chart_type == 'line':
          key_point_cnv = self.key_cnvs[ind](cnv)
          hybrid_cnv = self.hybrid_cnvs[ind](cnv)

          key_heat,hybrid_heat = self.key_heats[ind](key_point_cnv),self.hybrid_heats[ind](hybrid_cnv)
          key_tag_ori = self.key_tags[ind](cnv)
          key_regrs_ori = self.key_regrs[ind](key_point_cnv)

          # key_tag  = [_tranpose_and_gather_feature(e, inputs['key_tags']) for e in key_tag_ori]
          # key_regr = [_tranpose_and_gather_feature(e, inputs['key_tags']) for e in key_regrs_ori]

          key_tag  = _tranpose_and_gather_feature(key_tag_ori, inputs['key_tags'])
          key_regr = _tranpose_and_gather_feature(key_regrs_ori, inputs['key_tags']) 

          key_tag_grouped = []
          for g_id in range(16):
            key_tag_grouped.append(torch.unsqueeze(_tranpose_and_gather_feature(key_tag_ori, inputs['key_tags_grouped'][:,g_id,:]), 1))
          key_tag_grouped = torch.cat(key_tag_grouped, 1)

          outs.append([key_heat, hybrid_heat, key_tag, key_tag_grouped, key_regr])

        if self.chart_type == 'line_query':
          ps_features = _tranpose_and_gather_feature(cnv, inputs["ps_inds"])
          ng_features = _tranpose_and_gather_feature(cnv, inputs["ng_inds"])

          ps_features_group = self._group_features(ps_features, inputs["ps_weight"])
          ng_features_group = self._group_features(ng_features, inputs["ng_weight"])

          ps_prediction = self.cls(ps_features_group)
          ng_prediction = self.cls(ng_features_group)

          outs.append([ps_prediction, ng_prediction])


      if ind < self.nstack - 1:
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
        inter = self.relu(inter)
        inter = self.inters[ind](inter)


    return outs

  def test_loop(self,):
    return None

  def _group_features(self, features, weight):
      features = features.view(features.size(0), -1, 4, features.size(2))
      weight = weight.view(weight.size(0), -1, 4)
      weight = weight.unsqueeze(3)
      weighted_features = features * weight
      weighted_features = torch.sum(weighted_features, 2)
      weighted_features = weighted_features.view(weighted_features.size(0), -1, 8*weighted_features.size(2))
      return weighted_features

  def bar_chart_val(self,inputs):
    output = self.train_loop(inputs,for_val=False
    )
    K=100
    kernel = 1
    ae_threshold = 1
    num_dets =1000

    tl_heat, br_heat, tl_regr, br_regr = output[0][0],output[0][1],output[0][2],output[0][3]
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    # print(tl_scores)
    tl_regr_ = _tranpose_and_gather_feature(tl_regr, tl_inds)
    br_regr_ = _tranpose_and_gather_feature(br_regr, br_inds)


    tl_scores_ = tl_scores.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_clses_.float(), br_xs_, br_ys_], dim=0)

    return detections_tl, detections_br



# tiny hourglass is for f**king debug
def get_hourglass(hourglass_type,chart_type,for_inference):
  get_hourglass_dict = \
    {
    #'large_hourglass' : exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],c_type=chart_type,for_inference=for_inference),
    'large_hourglass' : exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],c_type=chart_type,for_inference=for_inference),
    'small_hourglass' : exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4],c_type=chart_type,for_inference=for_inference),
    'tiny_hourglass'  : exkp(n=5, nstack=1, dims=[256, 128, 256, 256, 256, 384], modules=[2, 2, 2, 2, 2, 4],c_type=chart_type,for_inference=for_inference)
    }
  return get_hourglass_dict[hourglass_type]
    

if __name__ == '__main__':
  import time
  import pickle
  from collections import OrderedDict


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_hourglass['tiny_hourglass'].cuda()

  # ckpt = torch.load('./ckpt/pretrain/checkpoint.t7', map_location='cpu')
  # new_ckpt = OrderedDict()
  # for k in ckpt:
  #   if 'up1' in k:
  #     new_ckpt[k.replace('up1', 'top')] = ckpt[k]
  #   elif 'tl_cnvs' in k:
  #     new_ckpt[k.replace('tl_cnvs', 'cnvs_tl')] = ckpt[k]
  #   elif 'br_cnvs' in k:
  #     new_ckpt[k.replace('br_cnvs', 'cnvs_br')] = ckpt[k]
  #   elif 'tl_heats' in k:
  #     new_ckpt[k.replace('tl_heats', 'hmap_tl')] = ckpt[k]
  #   elif 'br_heats' in k:
  #     new_ckpt[k.replace('br_heats', 'hmap_br')] = ckpt[k]
  #   elif 'tl_tags' in k:
  #     new_ckpt[k.replace('tl_tags', 'embd_tl')] = ckpt[k]
  #   elif 'br_tags' in k:
  #     new_ckpt[k.replace('br_tags', 'embd_br')] = ckpt[k]
  #   elif 'tl_regrs' in k:
  #     new_ckpt[k.replace('tl_regrs', 'regs_tl')] = ckpt[k]
  #   elif 'br_regrs' in k:
  #     new_ckpt[k.replace('br_regrs', 'regs_br')] = ckpt[k]
  #   else:
  #     new_ckpt[k] = ckpt[k]
  # torch.save(new_ckpt, './ckpt/pretrain/checkpoint.t7')

  # net.load_state_dict(ckpt)

  print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))

  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.register_forward_hook(hook)

  with torch.no_grad():
    y = net(torch.randn(1, 3, 384, 384).cuda())
  # print(y.size())
