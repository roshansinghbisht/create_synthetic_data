import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import pytz
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import random
import os
from PIL import Image
import numpy as np
import sys
from style_transfer import image_loader, run_style_transfer, cnn, cnn_normalization_mean,\
cnn_normalization_std
# ContentLoss, StyleLoss, get_style_model_and_losses,\
# get_input_optimizer,
    

class CreateSyntheticData():
    """Creates annotated data randomly
    """
    def __init__(self, output_dir, image_dir):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


        self.char2id = {c[1]:c[0]+1 for c in enumerate(self.chars)}
        self.coco = Coco()
        self.output_dir = output_dir
        self.image_dir = image_dir
        for c in self.chars:
            self.coco.add_category(CocoCategory(id=self.char2id[c], name=c))        


    def set_valid_dimensions(self):
        """Set valid image dimensions randomly based upon the aspect ratio
        """
        for i in range(100000):
            self.image_width = random.randint(300, 300)
            self.image_height = random.randint(300, 300)
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
        org_x = random.randint(5, 25)
        org_y = random.randint(50, self.image_height)
        if self.image_width >= 700:
            font_scale = 2#2.5
            thickness = 4
        elif self.image_width >=600 and self.image_width <700:
            font_scale = 1.5#2
            thickness = 3
        else:
            font_scale = 0.7
            thickness = 1
        self.config = {"font": cv2.FONT_HERSHEY_SIMPLEX, "color" : (0, 0, 0), "org": (org_x, org_y),
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
        ret, binary_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
                # x, y, w, h = cv2.boundingRect(c)
                # cv2.rectangle(image,(x,y), (x+w,y+h), (0,0,255), 1)
        # cv2.imshow('image with contours', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(self.image_dir, self.img_name), image)
        return contours


    def sort_contours(self, cnts):
        """Return the list of sorted contours and bounding boxes
        """
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
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


    def neural_style_transfer(self, content_img, style_img):
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(np.uint8(style_img)).convert('RGB')
        style_img = image_loader(style_img)
        content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        content_img = Image.fromarray(np.uint8(content_img)).convert('RGB')
        content_img = image_loader(content_img)
        input_img = content_img.clone()
        output_tensor = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)
        date = datetime.now().strftime("%Y_%m_%d %H:%M:%S.%f")
        self.img_name = f"created_image_{date}.jpg"
        tensor_clone = output_tensor.cpu().clone()  # we clone the tensor to not do changes on it
        tensor_clone = tensor_clone.squeeze(0)      # remove the fake batch dimension
        output_image = tensor_clone.permute(1, 2, 0).detach().numpy()
        output_image = output_image * 255
        cv2.imwrite(os.path.join(self.image_dir, self.img_name), output_image)


    def main(self, image, textured_image, style_image):
        new_res_image = self.change_resolution(image)
        new_res_textured_image = textured_image[:self.image_height, :self.image_width]
        new_res_style_image = style_image[:self.image_height, :self.image_width]
        self.initialize_parameters()
        self.chars_sample = random.sample(self.chars, 11)
        for char in self.chars_sample:
            new_res_image, char_size = self.put_character(new_res_image, char)
            new_res_textured_image, char_size = self.put_character(new_res_textured_image, char)
            self.config["org"] = (self.config["org"][0]+char_size[0][0], self.config["org"][1])
        self.neural_style_transfer(new_res_textured_image, new_res_style_image)
        contours = self.find_contours(new_res_image)
        self.create_coco_images(contours)

    
if __name__ == "__main__":
    train_split = sys.argv[1]
    test_split = sys.argv[2]    
    date_time = datetime.now().strftime("%Y_%m_%d %H_%M_%S.%f")
    output_dir_path = os.path.join("./created_data", f"output_{date_time}")
    os.mkdir(output_dir_path)
    train_split_path = os.path.join(output_dir_path, "train")
    os.mkdir(train_split_path)
    train_image_dir = os.path.join(train_split_path, "images")
    os.mkdir(train_image_dir)
    test_split_path = os.path.join(output_dir_path, "test")
    os.mkdir(test_split_path)
    test_image_dir = os.path.join(test_split_path, "images")
    os.mkdir(test_image_dir)


    csd_train = CreateSyntheticData(train_split_path, train_image_dir)
    for i in range(int(train_split)):
        image = cv2.imread('./images/paper_texture.jpg')
        texture_image = cv2.imread('./images/textured_image4.jpg')
        style_image = cv2.imread('./images/tyre_style4.png')
        csd_train.main(image, texture_image, style_image)
        csd_train.save()

    # csd_test = CreateSyntheticData(test_split_path, test_image_dir)
    # for i in range(int(test_split)):
    #     image = cv2.imread('./images/paper_texture.jpg')
    #     texture_image = cv2.imread('./images/textured_image4.jpg')
    #     style_image = cv2.imread('./images/tyre_style4.png')
    #     csd_test.main(image, texture_image, style_image)
    # csd_test.save()