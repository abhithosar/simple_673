import torch
import torch.nn as nn
from nets.hourglass import get_hourglass
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature
import numpy as np
import cv2
from utils.losses import _neg_loss, _ae_loss, _reg_loss, Loss


def get_bar_chart_model(hourglass_framework_type,for_inference):
    model = get_hourglass(hourglass_framework_type,'bar',for_inference)
    return model

def bar_chart_loss(preds,target):
    batch = target
    #hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*preds)
    hmap_tl, hmap_br, regs_tl, regs_br = zip(*preds)

    # embd_tl = [_tranpose_and_gather_feature(e, batch['inds_tl']) for e in embd_tl]
    # embd_br = [_tranpose_and_gather_feature(e, batch['inds_br']) for e in embd_br]
    # regs_tl = [_tranpose_and_gather_feature(r, batch['inds_tl']) for r in regs_tl]
    # regs_br = [_tranpose_and_gather_feature(r, batch['inds_br']) for r in regs_br]

    focal_loss = _neg_loss(hmap_tl, batch['hmap_tl']) + \
                   _neg_loss(hmap_br, batch['hmap_br'])

    reg_loss = _reg_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
                 _reg_loss(regs_br, batch['regs_br'], batch['ind_masks'])

    #pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ind_masks'])

    #loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
    #return (loss,(focal_loss,reg_loss,pull_loss,push_loss))
    loss = focal_loss + reg_loss
    return loss.unsqueeze(0)


def get_inference_on_bar(nnet,image):
    K = 100
    ae_threshold = 0.5
    nms_kernel = 3

    categories = 1

    nms_threshold = 0.5
    max_per_image = 100

    height,width = image.shape[0:2]

    detections_point_tl = []
    detections_point_br  =[]

    scale = 1.0

    new_height = int(height*scale)
    new_width = int(width*scale)
    
    new_center = np.array([new_height // 2, new_width // 2])

    inp_height = new_height | 127
    inp_width  = new_width  | 127
    images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios  = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes   = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio = out_height / inp_height
    width_ratio  = out_width  / inp_width

    resized_image = cv2.resize(image, (new_width, new_height))
    resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

    resized_image = resized_image / 255.
        #normalize_(resized_image, db.mean, db.std)

    images[0]  = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0]   = [int(height * scale), int(width * scale)]
    ratios[0]  = [height_ratio, width_ratio]

    if torch.cuda.is_available():
        images = torch.from_numpy(images).cuda(0)
    else:
        images = torch.from_numpy(images)
    dets_tl = None
    dets_br = None
    with torch.no_grad():
            inputs = {'image' : images[0]}
            detections_tl, detections_br = nnet(inputs)
            #detections_tl = detections_tl_detections_br[0]
            #detections_br = detections_tl_detections_br[1]
            dets_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
            dets_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
    #dets_tl, dets_br, flag = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
    offset = (offset + 1) * 100
    del images
    _rescale_points(dets_tl, ratios, borders, sizes)
    _rescale_points(dets_br, ratios, borders, sizes)
    detections_point_tl.append(dets_tl)
    detections_point_br.append(dets_br)
    detections_point_tl = np.concatenate(detections_point_tl, axis=1)
    detections_point_br = np.concatenate(detections_point_br, axis=1)

    classes_p_tl = detections_point_tl[:, 0, 1]
    classes_p_br = detections_point_br[:, 0, 1]

    keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
    detections_point_tl = detections_point_tl[keep_inds_p, 0]
    classes_p_tl = classes_p_tl[keep_inds_p]

    keep_inds_p = (detections_point_br[:, 0, 0] > 0)
    detections_point_br = detections_point_br[keep_inds_p, 0]
    classes_p_br = classes_p_br[keep_inds_p]

    top_points_tl = {}
    top_points_br = {}
    for j in range(categories):

        keep_inds_p = (classes_p_tl == j)
        top_points_tl[j + 1] = detections_point_tl[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_br == j)
        top_points_br[j + 1] = detections_point_br[keep_inds_p].astype(np.float32)
        #print(top_points[image_id][j + 1][0])


    scores = np.hstack([
        top_points_tl[j][:, 0]
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_tl[j][:, 0] >= thresh)
            top_points_tl[j] = top_points_tl[j][keep_inds]

    scores = np.hstack([
        top_points_br[j][:, 0]
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_br[j][:, 0] >= thresh)
            top_points_br[j] = top_points_br[j][keep_inds]

    return top_points_tl, top_points_br


def _rescale_points(dets, ratios, borders, sizes):
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    xs    /= ratios[0, 1]
    ys    /= ratios[0, 0]
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)


def crop_image(image, center, size):
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset
