import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import random
import os
from PIL import Image


class CreateSyntheticData():
    """Creates annotated data randomly
    """
    def __init__(self, image):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
        self.char2id = {c[1]:c[0] for c in enumerate(self.chars)}
        self.image = image


    def set_valid_dimensions(self):
        """Set valid image dimensions randomly based upon the aspect ratio
        """
        for i in range(100000):
            self.image_width = random.randint(400, 800)
            self.image_height = random.randint(400, 800)
            aspect_ratio =  self.image_width / self.image_height
            if aspect_ratio >= 1:
                break
    
    
    def change_resolution(self, image):
        """Change image resolution of the given image
        """
        self.set_valid_dimensions()
        new_res_image = image[:self.image_height, :self.image_width]
        
        return new_res_image
    

    def initialize_parameters(self, image):
        """ Creates the config parameters
        """
        org_x = random.randint(10, 50)
        org_y = random.randint(200, self.image_height)
        if self.image_width >= 700:
            font_scale = 2.5
            thickness = random.randint(6, 8)
        elif self.image_width >=600 and self.image_width <700:
            font_scale = 2
            thickness = 3
        else:
            font_scale = 1
            thickness = 2
        self.config = {"font": cv2.FONT_HERSHEY_SIMPLEX, "color" : (255, 0, 0), "org": (org_x, org_y),
                  "font_scale": font_scale, "thickness": thickness} 
        
        
    def put_character(self, image, character):
        """ Writes character on an image.
        """
        image = cv2.putText(image, character, self.config["org"], self.config["font"],
                            self.config["font_scale"], self.config["color"], self.config["thickness"],cv2.LINE_AA)
        char_size = cv2.getTextSize(character ,self.config["font"],
                            self.config["font_scale"], self.config["thickness"])

        return image, char_size


    def find_contours(self, image):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        inverted_binary = ~binary
        contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.image, (x,y), (x+w,y+h), (0,0,255), 1)
        cv2.imshow('Image with bounding boxes', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image, contours


    def create_coco_file(self, contours, image):
        coco = Coco()
        for c in self.chars:
            coco.add_category(CocoCategory(id=self.char2id[c], name=c))
        coco_image = CocoImage(file_name="out.jpg", height=self.image_height, width=self.image_width)
        date = datetime.now().strftime("%Y_%m_%d %H:%M:%S.%f")
        path = '/home/roshan/Desktop/EFleet/Synthetic_simple_data/created_images'
        cv2.imwrite(os.path.join(path , f"out_{date}.jpg"), image)
        for (contour, char) in zip(contours, self.chars_sample):
            x, y, w, h = cv2.boundingRect(contour)
            coco_image.add_annotation(CocoAnnotation(
                bbox=[x, y, w, h],category_id=self.char2id[char],
                category_name=char)
            )
        coco.add_image(coco_image)
        save_json(data=coco.json, save_path=f"/home/roshan/Desktop/EFleet/Synthetic_simple_data/created_json/coco_annotations{date}.json")
    
    
    def main(self):
        self.image = self.change_resolution(self.image)
        self.initialize_parameters(self.image)
        self.chars_sample = random.sample(self.chars, 11)
        for char in self.chars_sample:
            self.image, char_size = self.put_character(self.image, char)
            self.config["org"] = (self.config["org"][0]+char_size[0][0], self.config["org"][1])
        self.image, contours = self.find_contours(self.image)
        self.create_coco_file(contours, self.image)
    
    
if __name__ == "__main__":
    image = cv2.imread('paper_texture.jpg')
    # for i in range(2):
    csd = CreateSyntheticData(image)
    csd.main()
    
    