import torch
import torch.nn as nn
from nets.hourglass import get_hourglass
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature
import numpy as np
import cv2
from utils.losses import _ae_line_loss,_neg_loss, _ae_loss, _reg_loss, Loss, AELossLine

from utils.image import crop_image

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

def get_inference_on_line(nnet, image):
    K = 200#top_k
    ae_threshold = 0.5
    nms_kernel = 3

    categories = 1

    nms_threshold = 0.5
    max_per_image = 100

    time_backbones = 0
    time_psns = 0

    height, width = image.shape[0],image.shape[1]

    detections_point_key = []
    detections_point_hybrid = []
    if height > 1500 or width > 1500:
        scale = 0.3
    elif (height > 1000 and height < 1500) or (width > 1000 and width<1500):
        scale = 0.47
    else:
        scale = 1.0
    new_height = int(height * scale)
    new_width  = int(width * scale)
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
    with torch.no_grad():
        inputs = {'image' : images}
        detections_tl, detections_br = nnet(inputs)
        #detections_tl = detections_tl_detections_br[0]
        #detections_br = detections_tl_detections_br[1]
        dets_key = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
        dets_hybrid = detections_br.data.cpu().numpy().transpose((2, 1, 0))
        #return detections_tl, detections_br, time_backbone, time_psn, True
    # dets_key, dets_hybrid, time_backbone, time_psn, flag = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
    #time_backbones += time_backbone
   # time_psns += time_psn
    #print(dets_key.shape)
    _rescale_points(dets_key, ratios, borders, sizes)
    _rescale_points(dets_hybrid, ratios, borders, sizes)
    #print(dets_key)
    detections_point_key.append(dets_key)
    detections_point_hybrid.append(dets_hybrid)
    detections_point_key = np.concatenate(detections_point_key, axis=1)
    detections_point_hybrid = np.concatenate(detections_point_hybrid, axis=1)
    #print(detections_point_key[:, 0, 0])
    #print('1')
    #print(detections_point.shape)

    classes_p_key = detections_point_key[:, 0, 2]
    classes_p_hybrid = detections_point_hybrid[:, 0, 2]
    #print('2')
    #print(classes_p.shape)

    # reject detections with negative scores

    keep_inds_p = (detections_point_key[:, 0, 0] > 0)
    detections_point_key = detections_point_key[keep_inds_p, 0]
    classes_p_key = classes_p_key[keep_inds_p]

    keep_inds_p = (detections_point_hybrid[:, 0, 0] > 0)
    detections_point_hybrid = detections_point_hybrid[keep_inds_p, 0]
    classes_p_hybrid = classes_p_hybrid[keep_inds_p]

    #print('3')
    #print(detections_point.shape)

    top_points_key = {}
    top_points_hybrid = {}
    for j in range(categories):

        keep_inds_p = (classes_p_key == j)
        top_points_key[j + 1] = detections_point_key[keep_inds_p].astype(np.float32)
        keep_inds_p = (classes_p_hybrid == j)
        top_points_hybrid[j + 1] = detections_point_hybrid[keep_inds_p].astype(np.float32)
        #print(top_points[image_id][j + 1][0])


    scores = np.hstack([
        top_points_key[j][:, 0]
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_key[j][:, 0] >= thresh)
            top_points_key[j] = top_points_key[j][keep_inds]

    scores = np.hstack([
        top_points_hybrid[j][:, 0]
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_points_hybrid[j][:, 0] >= thresh)
            top_points_hybrid[j] = top_points_hybrid[j][keep_inds]
    quiry, keys, hybrids = GroupQuiryRaw(top_points_key, top_points_hybrid)
    return GroupLineRaw(keys, hybrids)



def _rescale_points(dets, ratios, borders, sizes):
    xs, ys = dets[:, :, 3], dets[:, :, 4]
    xs    /= ratios[0, 1]
    ys    /= ratios[0, 0]
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)



def GroupLineRaw(keys, hybrids):
    union_result = group_points(keys)
    # hybrid_points = [key for key in keys if key['is_cross']]
    # union_result = try_match(union_result, hybrid_points, pair_info)
    #image.save(tar_dir + id2name[id])
    data_points = []
    for line in union_result:
        data_line = []
        if len(line) > 1:
            #draw_group(line, image)
            for point in line:
                if point is not None:
                    data_line.append([point['bbox'][0], point['bbox'][1]])
            data_points.append(data_line)
    return data_points


def get_key(a):
    return a[1]
def group_points(keys):
    dis_array = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            dis_array[(i, j)] = compute_tag_dis(keys[i], keys[j])
    dis_array = list(dis_array.items())
    dis_array.sort(key=get_key)
    unionset = UnionFindSet(keys)
    for pair in dis_array:
        if pair[1] > 0.125:#threshold_tag:
            break
        unionset.union(pair[0][0], pair[0][1])
    group = {}
    for i in range(len(keys)):
        if unionset.size_dict[i] > 0:
            group[i] = []
    for i in range(len(keys)):
        group[unionset.father_dict[i]].append(i)
    #print(group)
    group = list(group.values())
    for line in  group:
        for i in range(len(line)):
            line[i] = keys[line[i]]
    return group

def compute_tag_dis(key1, key2):
    return abs(key1['tag']- key2['tag'])


class UnionFindSet(object):
    def __init__(self, data_list):
        self.father_dict = {}
        self.size_dict = {}
        for i in range(len(data_list)):
            self.father_dict[i] = i
            self.size_dict[i] = 1

    def find_head(self, ID):

        father = self.father_dict[ID]
        if(ID != father):
            father = self.find_head(father)
        self.father_dict[ID] = father
        return father

    def is_same_set(self, ID_a, ID_b):
        return self.find_head(ID_a) == self.find_head(ID_b)

    def union(self, ID_a, ID_b):
        if ID_a is None or ID_a is None:
            return

        a_head = self.find_head(ID_a)
        b_head = self.find_head(ID_b)

        if(a_head != b_head):
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if(a_set_size >= b_set_size):
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size

def GroupQuiryRaw(keys_raw, hybrids_raw):
    keys = []
    for temp in keys_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])

            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    hybrids = []
    for temp in hybrids_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])
            hybrids.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    keys = get_point(keys, 0.4)
    hybrids = get_point(hybrids, 0.4)
    keys = check_cross(keys, hybrids)
    quiries = quiry_for_hybrid(keys)
    return quiries, keys, hybrids


def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point['score'] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean


def check_cross(keys, hybrids):
    for key in keys:
        key['is_cross'] = False
    for hybrid in hybrids:
        border = [hybrid['bbox'][0]-4, hybrid['bbox'][1]-4, hybrid['bbox'][0]+4, hybrid['bbox'][1]+4]
        for key in keys:
            if key['bbox'][0] >= border[0] and key['bbox'][1] >= border[1] and key['bbox'][0] <= border[2] and key['bbox'][1] <= border[3]:
                key['is_cross'] = True
    return keys


def quiry_for_hybrid(keys):
    keys.sort(key = lambda x:x['bbox'][0])
    quirys = []
    for ind, key in enumerate(keys):
        if key['is_cross']:
            rp_ind_s = ind+1
            for rp_ind_s in range(ind+1, len(keys)):
                if abs(keys[rp_ind_s]['bbox'][0]-key['bbox'][0])>4:
                    break
            rp_ind_e = min(rp_ind_s + 1, len(keys))
            for rp_ind_e in range(rp_ind_s, len(keys)):
                if abs(keys[rp_ind_e]['bbox'][0]-keys[rp_ind_s]['bbox'][0])>4:
                    break
            for r_ind in range(rp_ind_s, rp_ind_e):
                quiry_pair = [[], []]
                quiry_pair[0] = [key['bbox'][0], key['bbox'][1]]
                quiry_pair[1] = [keys[r_ind]['bbox'][0], keys[r_ind]['bbox'][1]]
                quirys.append(quiry_pair)
            lp_ind_s = ind - 1
            for lp_ind_s in range(ind - 1, -1, -1):
                if abs(keys[lp_ind_s]['bbox'][0] - key['bbox'][0])>4:
                    break
            lp_ind_e = max(lp_ind_s - 1, -1)
            for lp_ind_e in range(lp_ind_s, -1, -1):
                if abs(keys[lp_ind_e]['bbox'][0] - keys[lp_ind_s]['bbox'][0])>4:
                    break
            for l_ind in range(lp_ind_s, lp_ind_e, -1):
                quiry_pair = [[], []]
                quiry_pair[1] = [key['bbox'][0], key['bbox'][1]]
                quiry_pair[0] = [keys[l_ind]['bbox'][0], keys[l_ind]['bbox'][1]]
                quirys.append(quiry_pair)
    return quirys
