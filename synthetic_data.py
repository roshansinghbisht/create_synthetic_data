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
import numpy as np
# import argparse
# import imutils


class CreateSyntheticData():
    """Creates annotated data randomly
    """
    def __init__(self, output_dir, image_dir):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


        self.char2id = {c[1]:c[0] for c in enumerate(self.chars)}
        # self.image = image
        self.coco = Coco()
        self.output_dir = output_dir
        self.image_dir = image_dir
        for c in self.chars:
            self.coco.add_category(CocoCategory(id=self.char2id[c], name=c))        


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
    

    def initialize_parameters(self):
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # inverted_binary = ~binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if (cv2.contourArea(c)) > 50:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image,(x,y), (x+w,y+h), (0,0,255), 2)
        return contours


    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return cnts
    

    def create_coco_images(self, contours):
        coco_image = CocoImage(file_name=self.img_name, height=self.image_height, width=self.image_width)
        contours = self.sort_contours(contours)
        for (contour, char) in zip(contours, self.chars_sample):
            x, y, w, h = cv2.boundingRect(contour)
            coco_image.add_annotation(CocoAnnotation(
                bbox=[x, y, w, h],category_id=self.char2id[char],
                category_name=char)
            )
        self.coco.add_image(coco_image)


    def save(self):
        save_json(data=self.coco.json, save_path=f"{self.output_dir}/output.json")

    
    def main(self, image):
        new_res_image = self.change_resolution(image)
        self.initialize_parameters()
        self.chars_sample = random.sample(self.chars, 11)
        for char in self.chars_sample:
            new_res_image, char_size = self.put_character(new_res_image, char)
            self.config["org"] = (self.config["org"][0]+char_size[0][0], self.config["org"][1])
        date = datetime.now().strftime("%Y_%m_%d %H:%M:%S.%f")
        self.img_name = f"created_image_{date}.jpg"
        cv2.imwrite(os.path.join(self.image_dir, self.img_name), new_res_image)
        contours = self.find_contours(new_res_image)
        self.create_coco_images(contours)

    
if __name__ == "__main__":
    output_dir = datetime.now().strftime("%Y_%m_%d %H_%M_%S.%f")
    output_dir_path = os.path.join(".", f"output_{output_dir}")
    os.mkdir(output_dir_path)
    image_dir_name = "images"
    image_dir_path = os.path.join(output_dir_path, image_dir_name)
    os.mkdir(image_dir_path)

    csd = CreateSyntheticData(output_dir_path, image_dir_path)
    for i in range(15):
        image = cv2.imread('paper_texture.jpg')
        csd.main(image)
    csd.save()