# coding=utf-8
import xml.etree.ElementTree as ET
import os
import random

from yolo_config import task, task_yolo_config

output_dir = "data/{}".format(task)
train_file_path = task_yolo_config.data_config["train"]
valid_file_path = task_yolo_config.data_config["valid"]

image_dir = task_yolo_config.image_dir
voc_dir = task_yolo_config.voc_dir
classes = task_yolo_config.classes


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_file_name):
    image_id = image_file_name.replace(".jpg", "").replace(".png", "")
    voc_file_name = "{}.xml".format(image_id)
    voc_file_path = os.path.join(voc_dir, voc_file_name)

    if not os.path.exists(voc_file_path):
        return None

    txt_file_name = "{}.txt".format(image_id)
    txt_file_path = os.path.join(image_dir, txt_file_name)

    image_file_path = os.path.join(image_dir, image_file_name)

    with open(voc_file_path, "r", encoding="utf-8") as voc_file, \
            open(txt_file_path, "w", encoding="utf-8") as txt_file:

        tree = ET.parse(voc_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            line = "{} {}".format(cls_id, " ".join([str(a) for a in bb]))
            txt_file.write("{}\n".format(line))
            print("{} {}".format(image_file_path, line))
    return txt_file_path


with open(train_file_path, "w", encoding="utf-8") as train_file, \
        open(valid_file_path, "w", encoding="utf-8") as valid_file:
    image_file_names = [image_file_name for image_file_name in os.listdir(image_dir) if
                        not image_file_name.endswith("txt")]
    for image_file_name in image_file_names:
        txt_file_path = convert_annotation(image_file_name)
        if txt_file_path is not None:
            image_file_path = os.path.join(image_dir, image_file_name)
            if random.randrange(0, 10) < 10:
                train_file.write("{}\n".format(image_file_path))
            else:
                valid_file.write("{}\n".format(image_file_path))
