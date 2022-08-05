import numpy as np
from PIL import Image
import requests
from pycocotools.coco import COCO
import json
import sys
import re
import os
import random
import cv2
from datetime import datetime
import albumentations as A
import shutil


class data_augmentation():
    """Creates annotated data randomly
    """
    def __init__(self, base_img_folder, base_ann_file_path, sqr_crop_img_folder,
                 sqr_crop_ann_folder):
        self.base_image_folder = base_img_folder
        self.base_ann_file_path = base_ann_file_path
        self.sqr_crop_img_folder = sqr_crop_img_folder
        self.sqr_crop_ann_folder = sqr_crop_ann_folder
        shutil.copy(self.base_ann_file_path, self.sqr_crop_ann_folder)

        self.coco_annotation = COCO(annotation_file=self.base_ann_file_path) 
        self.img_ids = self.coco_annotation.getImgIds()
        self.ann_ids = self.coco_annotation.getAnnIds(self.img_ids)
        self.anns = self.coco_annotation.loadAnns(self.ann_ids)


    def get_dims(self, tag_x, tag_y, tag_w, tag_h, img_width, img_height):
        start_x = random.randint(0, int(tag_x))
        start_y = random.randint(0, int(tag_y))
        low_w = 1+(tag_x-start_x)+tag_w
        upper_w = img_width-start_x
        width = random.randint(int(low_w), int(upper_w))
        low_h = 1+(tag_y-start_y)+tag_h
        upper_h = img_height-start_y
        height = random.randint(int(low_h), int(upper_h))
        end_x = start_x + width
        end_y = start_y + height

        return start_x, start_y, end_x, end_y, width, height   
              

    def square_crop(self):
        """
        """
        with open(self.sqr_crop_ann_folder+"/result.json", "r") as jsonFile:
            data = json.load(jsonFile)
        new_image_data = []
        new_annotations_data = []

        for i in self.img_ids:
            bbox = [ann['bbox'] for ann in self.anns if ann['segmentation'] and ann['image_id'] == i]
            tag_x, tag_y, tag_w, tag_h = bbox[0]
            img_info = self.coco_annotation.loadImgs(i)[0]
            image_w = img_info["width"]
            image_h = img_info["height"]
            image_name = img_info["file_name"]
            start_x, start_y, end_x, end_y, cropped_width, cropped_height = self.get_dims(tag_x, tag_y, tag_w, tag_h, image_w, image_h)
            img = cv2.imread(self.base_image_folder+"/"+image_name)
            cropped_image = img[start_y:end_y, start_x:end_x]
            corr_x = -start_x
            corr_y = -start_y
            if cropped_width > cropped_height:
                new_dim = cropped_width
                diff = cropped_width-cropped_height
                if diff%2 !=0:
                    image = cv2.copyMakeBorder(cropped_image, (diff-1)//2, (diff+1)//2, 0, 0, cv2.BORDER_REPLICATE, None, value = 0)
                    corr_y += (diff-1)//2
                else:
                    image = cv2.copyMakeBorder(cropped_image, diff//2, diff//2, 0, 0, cv2.BORDER_REPLICATE, None, value = 0)
                    corr_y += diff//2
            elif cropped_height > cropped_width:
                new_dim = cropped_height
                diff = cropped_height-cropped_width
                if diff%2 !=0:
                    image = cv2.copyMakeBorder(cropped_image, 0, 0, (diff-1)//2, (diff+1)//2, cv2.BORDER_REPLICATE, None, value = 0)
                    corr_x += (diff-1)//2
                else:
                    image = cv2.copyMakeBorder(cropped_image, 0, 0, diff//2, diff//2, cv2.BORDER_REPLICATE, None, value = 0)
                    corr_x += diff//2
            else:
                pass
            date = datetime.now().strftime("%m_%d %H:%M:%S.%f")
            updated_name = f"aug_img_{image_name}_{date}.jpg"
            cv2.imwrite(os.path.join(self.sqr_crop_img_folder, updated_name), image)
            image_info = data["images"][i]
            image_info["width"] = new_dim
            image_info["height"] = new_dim
            image_info["file_name"] = updated_name
            new_image_data.append(image_info)

            for anno_info in data['annotations']:
                if anno_info['image_id'] == i:
                    anno_info['bbox'][0] += corr_x
                    anno_info['bbox'][1] += corr_y
                    if anno_info['segmentation']:
                        for j in range(len(anno_info['segmentation'][0])):
                            if j%2 == 0:
                                anno_info['segmentation'][0][j] += corr_x
                            else:
                                anno_info['segmentation'][0][j] += corr_y
                    new_annotations_data.append(anno_info)  

        data['images'] = new_image_data
        data['annotations'] = new_annotations_data
        with open(self.sqr_crop_ann_folder+"/result.json", 'w') as jsonFile:
            json.dump(data, jsonFile) 


    # def color_jitter(self, image):
    #     '''
        
    #     '''
    #     transform = A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.8, hue=0.9, always_apply=False, p = 0.5)
    #     jittered_image = transform(image=image)['image']
    #     return jittered_image


    # def sharpen(self, image):
    #     '''
        
    #     '''
    #     transform = A.Sharpen()
    #     sharpened_image = transform(image=image)['image']
    #     return sharpened_image
    

    # def noise(img, noise_type="gauss"):
    #     '''
    #     ### Adding Noise ###
    #     img: image
    #     cj_type: {gauss: gaussian, sp: salt & pepper}
    #     '''
    #     if noise_type == "gauss":
    #         image=img.copy() 
    #         mean=0
    #         st=0.7
    #         gauss = np.random.normal(mean,st,image.shape)
    #         gauss = gauss.astype('uint8')
    #         image = cv2.add(image,gauss)
    #         return image
    
    #     elif noise_type == "sp":
    #         image=img.copy() 
    #         prob = 0.05
    #         if len(image.shape) == 2:
    #             black = 0
    #             white = 255            
    #         else:
    #             colorspace = image.shape[2]
    #             if colorspace == 3:  # RGB
    #                 black = np.array([0, 0, 0], dtype='uint8')
    #                 white = np.array([255, 255, 255], dtype='uint8')
    #             else:  # RGBA
    #                 black = np.array([0, 0, 0, 255], dtype='uint8')
    #                 white = np.array([255, 255, 255, 255], dtype='uint8')
    #         probs = np.random.random(image.shape[:2])
    #         image[probs < (prob / 2)] = black
    #         image[probs > 1 - (prob / 2)] = white
    #         return image



        def zoom_in(self, image, start_y, end_y, start_x, end_x):

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            cropped = image[start_y:end_y, start_x:end_x]
            zoomed_and_cropped = cv2.resize(cropped, (h, w) ,interpolation = cv2.INTER_LINEAR )


    def main(self):
        self.square_crop()
        
         


if __name__ == "__main__":
    base_img_folder = "base_images/images"
    base_ann_file_path = "base_images/result.json"
    sqr_crop_img_folder = "square_crop/images"
    sqr_crop_ann_folder = "square_crop/"
    zoomed_in_img_folder = "zoomed_in/images"
    zoomed_in_ann_folder = "zoomed_in/"
    vaf = data_augmentation(base_img_folder, base_ann_file_path, sqr_crop_img_folder, 
            sqr_crop_ann_folder)
    vaf.main()