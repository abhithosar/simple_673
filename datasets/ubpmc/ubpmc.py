
from torch.functional import split
import sys



#intellisense error will be there, file is loaded from path given
sys.path.insert(1,"utils/")
from image import color_jittering_,draw_gaussian,\
        gaussian_radius,lighting_,\
            random_crop, crop_image,\
                color_jittering_,lighting_,\
                    draw_gaussian, gaussian_radius

#region Torch Imports
import json
import torch
from torch.utils.data import Dataset

import numpy as np
import os
import glob
import cv2
import importlib
import math
#endregion

#region COCO Dataset related IDS and STATIC variables

#endregion
COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]
class UBPMCDataset_Bar(Dataset):
    def __init__(self, data_dir, is_Training =False, 
                       split_ratio=0.8, gaussian=True, 
                       img_size=511, dataset=None, arch=None, is_inference=False,testdb='ubpmc'):
        #self.split = dataset_split
        self.num_classes = 1

        self.is_training = is_Training
        self.is_inference = is_inference
        self.gaussian = gaussian
        self.down_ratio = 4
        self.image_size = {'h':img_size,'w':img_size}
        self.feature_map_size = {'h': (img_size + 1) // self.down_ratio, 'w': (img_size + 1) // self.down_ratio}

        self.data_range = np.random.RandomState(123)
        self.random_scales = np.arange(0.6,1.4,0.1)
        self.gaussian_iou = 0.3
        self.padding = 128

        dataset_split = None
        if 'bar' in arch:
            if is_Training:
                dataset_split = dataset['ubpmc_train_setup_bar']['train']
            else:
                dataset_split = dataset['ubpmc_train_setup_bar']['train_val']
            if is_inference:
                if 'ubpmc' in testdb:
                    dataset_split = dataset['ubpmc_train_setup_bar']['test_pmc']
                else:
                    dataset_split = dataset['ubpmc_train_setup_bar']['train_synth']
        

        self.data_dir = os.path.join(data_dir, 'ubpmc')
        
        classname = 'horizontal_bar'
        #TODO Generalized this path assignment for chart type
        self.img_dir = os.path.join(self.data_dir, "images",classname)
       
        annotations_folder = os.path.join(self.data_dir, "annotations_JSON", classname, "*.json")
        
        self.all_annotations = dict()
        all_files = glob.glob(annotations_folder)
        # for file in all_files:
        for file in dataset_split:
            data = dict()
            filename = file[1]#os.path.join(self.img_dir, os.path.splitext(os.path.basename(file))[0] + ".jpg")
            
            with open(file[0]) as f:
                data = json.load(f)
                if 'task6_output' in data and data['task6_output'] is not None:
                    bboxes = []
                    for bbox in data['task6_output']['output']['visual elements']['bars']:
                        x0 = bbox['x0']
                        y0 = bbox['y0']
                        x1 = x0 + bbox['width']
                        y1 = y0 + bbox['height']
                        bboxes.append([x0,y0,x1,y1])
                    self.all_annotations[filename] =  [np.asfarray(bboxes),file[0]]
                elif 'task6' in data and data['task6'] is not None:
                    bboxes = []
                    for bbox in data['task6']['output']['visual elements']['bars']:
                        x0 = bbox['x0']
                        y0 = bbox['y0']
                        x1 = x0 + bbox['width']
                        y1 = y0 + bbox['height']
                        bboxes.append([x0,y0,x1,y1])
                    self.all_annotations[filename] = [np.asfarray(bboxes),file[0]]
               
                f.close()
        split = int(len(self.all_annotations.keys())*split_ratio)
       # if self.is_training:
        self.images = list(self.all_annotations.keys())
        #else:
           # self.images = list(self.all_annotations.keys())[split:len(self.all_annotations.keys())]


        self.max_objs = 128
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.num_samples = len(self.images)

    def __getitem__(self, index: int):
        dsf = 9

        image_id = self.images[index]   
        image = np.asfarray(cv2.imread(image_id))
        
        bboxes = self.all_annotations[image_id][0]
        json_gt = self.all_annotations[image_id][1]
        if self.is_inference:
            return {'image': image, 'image_id':image_id, 'anno':bboxes, 'anno_file':json_gt}

        if(self.is_training):

            image, bboxes = random_crop(image,
            bboxes,
            random_scales=self.random_scales,
            new_size=self.image_size,
            padding=self.padding)
        else:
            image, border, offset = crop_image(image,
                        center=[image.shape[0] // 2, image.shape[1] // 2],
                        new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
            bboxes[:, 0::2] += border[2]
            bboxes[:, 1::2] += border[0]

        #Resize the image and bbox
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size['w'], self.image_size['h']))
        bboxes[:, 0::2] *= self.image_size['w'] / width
        bboxes[:, 1::2] *= self.image_size['h'] / height

        # discard non-valid bboxes
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.image_size['w'] - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.image_size['h'] - 1)
        keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
                               (bboxes[:, 3] - bboxes[:, 1]) > 0)

        bboxes = bboxes[keep_inds]
        image = image.astype(np.float32) / 255.
         # randomly flip image and bboxes
        if self.is_training and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]] - 1
        
        if self.is_training:
            color_jittering_(self.data_range, image)
            lighting_(self.data_range, image, 0.1, self.eig_val, self.eig_vec)

        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

        hmap_tl = np.zeros((self.num_classes, self.feature_map_size['h'], self.feature_map_size['w']), dtype=np.float32)
        hmap_br = np.zeros((self.num_classes, self.feature_map_size['h'], self.feature_map_size['w']), dtype=np.float32)

        regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)
        regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)
    
        inds_tl = np.zeros((self.max_objs,), dtype=np.int64)
        inds_br = np.zeros((self.max_objs,), dtype=np.int64)

        num_objs = np.array(min(bboxes.shape[0], self.max_objs))
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        ind_masks[:num_objs] = 1

        for (xtl, ytl, xbr, ybr) in bboxes:
            fxtl = (xtl * self.feature_map_size['w'] / self.image_size['w'])
            fytl = (ytl * self.feature_map_size['h'] / self.image_size['h'])
            fxbr = (xbr * self.feature_map_size['w'] / self.image_size['w'])
            fybr = (ybr * self.feature_map_size['h'] / self.image_size['h'])

            ixtl = int(fxtl)
            iytl = int(fytl)
            ixbr = int(fxbr)
            iybr = int(fybr)

            if self.gaussian:
                width = xbr - xtl
                height = ybr - ytl

                width = math.ceil(width * self.feature_map_size['w'] / self.image_size['w'])
                height = math.ceil(height * self.feature_map_size['h'] / self.image_size['h'])

                radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))

                draw_gaussian(hmap_tl[0], [ixtl, iytl], radius)
                draw_gaussian(hmap_br[0], [ixbr, iybr], radius)
            else:
                hmap_tl[0, iytl, ixtl] = 1
                hmap_br[0, iybr, ixbr] = 1

            regs_tl[0, :] = [fxtl - ixtl, fytl - iytl]
            regs_br[0, :] = [fxbr - ixbr, fybr - iybr]
            inds_tl[0] = iytl * self.feature_map_size['w'] + ixtl
            inds_br[0] = iybr * self.feature_map_size['w'] + ixbr

        return {'image': image,
            'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
            'regs_tl': regs_tl, 'regs_br': regs_br,
            'inds_tl': inds_tl, 'inds_br': inds_br,
            'ind_masks': ind_masks}

    def __len__(self) -> int:
        return self.num_samples

#class UBPMCDataset_Eval(UBPMCDataset_Bar):

class UBPMCDataset_Line(Dataset):
    def __init__(self,data_dir,is_Training =False, split_ratio=1.0, gaussian=True, img_size=511,
    dataset=None, arch=None, is_inference=False,testdb='ubpmc'):
        
        self.num_classes = 1
        self.is_inference = is_inference
        self.is_training = is_Training
        self.gaussian = gaussian
        self.down_ratio = 4
        self.image_size = {'h':img_size,'w':img_size}
        self.feature_map_size = {'h': (img_size + 1) // self.down_ratio, 'w': (img_size + 1) // self.down_ratio}

        self.data_range = np.random.RandomState(123)
        self.random_scales = np.arange(0.6,1.4,0.1)
        self.gaussian_iou = 0.3
        self.padding = 128

        dataset_split = None
        if 'line' in arch:
            if is_Training:
                dataset_split = dataset['ubpmc_train_setup_line']['train']
            else:
                dataset_split = dataset['ubpmc_train_setup_line']['train_val']
            if is_inference:
                if 'ubpmc' in testdb:
                    dataset_split = dataset['ubpmc_train_setup_line']['test_pmc']
                else:
                    dataset_split = dataset['ubpmc_train_setup_line']['train_synth']
        

        self.data_dir = os.path.join(data_dir, 'ubpmc')
        classname = 'line'
        self.img_dir = os.path.join(self.data_dir, "images",classname)
        annotations_folder = os.path.join(self.data_dir, "annotations_JSON", classname, "*.json")

        self.all_annotations = dict()
        all_files = glob.glob(annotations_folder)

        #for file in all_files:
        for file in dataset_split:
            data = dict()
            filename = file[1]#os.path.join(self.img_dir, os.path.splitext(os.path.basename(file))[0] + ".jpg")
            
            #with open(file) as f:
            with open(file[0]) as f:
                data = json.load(f)
                if 'task6_output' in data and data['task6_output'] is not None:
                    bboxes = []
                    for bbox in data['task6_output']['output']['visual elements']['lines']:
                        line =[]
                        for linept in linepts:
                            line.append(linept['x'])        
                            line.append(linept['y'])        
                        bboxes.append(line)
                    self.all_annotations[filename] =  np.asfarray(bboxes)
                elif 'task6' in data and data['task6'] is not None:
                    bboxes = []
                    for linepts in data['task6']['output']['visual elements']['lines']:
                        line =[]
                        for linept in linepts:
                            line.append(linept['x'])        
                            line.append(linept['y'])        
                        bboxes.append(line)
                    self.all_annotations[filename] = [np.asfarray(self.pad_to_dense(bboxes)),file[0]]
               
                f.close()
        self.images = list(self.all_annotations.keys())#[0:5]



        self.max_objs = 128
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.num_samples = len(self.images)

    def pad_to_dense(self,M):
        """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

        maxlen = max(len(r) for r in M)

        Z = np.zeros((len(M), maxlen))
        for enu, row in enumerate(M):
            Z[enu, :len(row)] += row 
        return Z

    def __getitem__(self, index: int):
        
        image_id = self.images[index]
        image = np.asfarray(cv2.imread(image_id))

        bboxes = self.all_annotations[image_id][0]
        json_gt = self.all_annotations[image_id][1]
        if self.is_inference:
            return {'image': image, 'image_id':image_id, 'anno':bboxes, 'anno_file':json_gt}

        if self.is_training:

            image, bboxes = random_crop(image,
            bboxes,
            random_scales=self.random_scales,
            new_size=self.image_size,
            padding=self.padding,isline=False)
        else:
            
            height, width = image.shape[0:2]

            max_hw = max(height, width)
            center = [height // 2, width // 2]
            size = [max_hw, max_hw]
            image, border, offset = crop_image(image, center, size)
            bboxes[:, 0::2] += border[2]
            bboxes[:, 1::2] += border[0]
        

        #Resize the image and bbox
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size['w'], self.image_size['h']))
        bboxes[:, 0::2] *= self.image_size['w'] / width
        bboxes[:, 1::2] *= self.image_size['h'] / height


        # discard non-valid bboxes
        bboxes[:, 0:bboxes.shape[1]:2] = np.clip(bboxes[:, 0:bboxes.shape[1]:2], 0, self.image_size['w'] - 1)
        bboxes[:, 1:bboxes.shape[1]:2] = np.clip(bboxes[:, 1:bboxes.shape[1]:2], 0, self.image_size['h'] - 1)
        # keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
        #                        (bboxes[:, 3] - bboxes[:, 1]) > 0)
        # bboxes = bboxes[keep_inds]

        if self.is_training and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]

        if self.is_training:
            image = image.astype(np.float32)/255.
            color_jittering_(self.data_range, image)
            lighting_(self.data_range, image, 0.1, self.eig_val, self.eig_vec)


        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

        max_tag_len = 256
        max_tag_len_group = 128
        max_group_len = 16
        
        key_heatmaps = np.zeros((self.num_classes, self.feature_map_size['h'], self.feature_map_size['w']), dtype=np.float32)
        hybrid_heatmaps = np.zeros((self.num_classes, self.feature_map_size['h'], self.feature_map_size['w']), dtype=np.float32)
      
        key_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
        key_tags = np.zeros((max_tag_len), dtype=np.int64)
        key_tags_grouped = np.zeros((max_group_len, max_tag_len_group), dtype=np.int64)
        tag_masks = np.zeros((max_tag_len), dtype=np.uint8)
        tag_masks_grouped = np.zeros(( max_group_len, max_tag_len_group), dtype=np.uint8)
        hybrid_masks_grouped = np.zeros(( max_group_len, max_tag_len_group), dtype=np.uint8)
        tag_lens = np.zeros((1,), dtype=np.int32)
        tag_group_lens = np.zeros((1,), dtype=np.int32)


        width_ratio =self.feature_map_size['w'] / self.image_size['w']
        height_ratio = self.feature_map_size['h'] / self.image_size['h']

        for ind,detection in enumerate(bboxes):

            fdetection = detection.copy()
            fdetection[0:len(fdetection):2] = detection[0:len(detection):2] * width_ratio
            fdetection[1:len(fdetection):2] = detection[1:len(detection):2] * height_ratio
            detection = fdetection.astype(np.int32)

            if self.gaussian:
                scale = np.random.choice(self.random_scales)
                width = image.shape[1] / 50 / 4 / scale
                height = image.shape[0] / 50 / 4 / scale

                radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))


                for k in range(int(len(detection) / 2)):
                    if not (detection[2*k] == 0 or detection[2*k+1] == 0 or detection[2*k] >= (self.feature_map_size['w']-1e-2) or detection[2*k+1] >= (self.feature_map_size['h']-1e-2)):
                        if key_heatmaps[0, detection[2 * k + 1], detection[2 * k]] < 0.85:
                            draw_gaussian(key_heatmaps[0], [detection[2 * k], detection[2 * k + 1]], radius)
                        else:
                            draw_gaussian(key_heatmaps[0], [detection[2 * k], detection[2 * k + 1]], radius)
                            draw_gaussian(hybrid_heatmaps[0], [detection[2 * k], detection[2 * k + 1]],
                                          radius)
            else:
                key_heatmaps[0, detection[2 * k + 1], detection[2 * k]] = 1

            
            for k in range(int(len(detection) / 2)):
                if not (detection[2 * k] == 0 or detection[2 * k + 1] == 0 or detection[2*k] >= (self.feature_map_size['w']-1e-2) or detection[2*k+1] >= (self.feature_map_size['w']-1e-2)):
                    if tag_lens[0] >= max_tag_len - 1 or k > max_tag_len_group - 1:
                        print("Too many targets, skip!")
                        print(tag_lens[0])
                        #print(image_file)
                        break
                    tag_ind = tag_lens[0]
                    key_regrs[tag_ind, :] = [fdetection[2 * k] - detection[2 * k],
                                                    fdetection[2 * k + 1] - detection[2 * k + 1]]
                    key_tags[tag_ind] = detection[2 * k + 1] * self.feature_map_size['w'] + detection[2 * k]
                    key_tags_grouped[ ind, k] = detection[2 * k + 1] * self.feature_map_size['w'] + detection[2 * k]
                    tag_lens[0] += 1
                    if hybrid_heatmaps[0, detection[2 * k + 1], detection[2 * k]] < 0.85:
                        tag_masks_grouped[ind, k] = 1
                    # print("Pre_tag_ing:%d" %tag_ind)
            tag_len = tag_lens[0]
            tag_group_lens[0] += 1
            tag_masks[:tag_len] = 1

            return {
                'image' : image,
                'key_heatmap':key_heatmaps,
                'key_regrs':key_regrs,
                'key_tags':key_tags,
                'tag_masks':tag_masks,
                'key_tags_grouped':key_tags_grouped,
                'tag_group_lens' : tag_group_lens,
                'hybrid_heatmaps' : hybrid_heatmaps,
                'tag_masks_grouped' : tag_masks_grouped
            }

    def __len__(self) -> int:
        return self.num_samples


if __name__ == '__main__':
    #import_parents(level=3)
    # ubpmc = UBPMCDataset_Bar("data",is_Training=True)
    ubpmc = UBPMCDataset_Line("data",is_Training=True)
    ubpmc.__getitem__(32)
    sdf = 0