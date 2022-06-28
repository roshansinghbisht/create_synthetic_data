from coco_assistant import COCO_Assistant
import os


def main():
    img_dir = os.path.join(os.getcwd(), 'images')
    ann_dir = os.path.join(os.getcwd(), 'annotations')
    cas = COCO_Assistant(img_dir, ann_dir)
    cas.merge()


if __name__ == "__main__":
    main()

