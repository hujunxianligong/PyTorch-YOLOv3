# coding=utf-8
import os
import torch
from utils.parse_config import parse_data_config
from utils.utils import load_classes

# use GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# configurations for YOLO
class YOLOConfig(object):
    def __init__(self,
                 epochs=201,
                 batch_size=16,
                 model_config_path="config/yolov3.cfg",
                 data_config_path="config/street.data",
                 weights_path="weights/yolov3.weights",
                 conf_thres=0.8,
                 nms_thres=0.4,
                 n_cpu=0,
                 img_size=416,
                 checkpoint_interval=5,
                 checkpoint_dir="checkpoints",
                 use_cuda=True
                 ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_config_path = model_config_path
        self.data_config_path = data_config_path
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.n_cpu = n_cpu
        self.img_size = img_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.use_cuda = use_cuda


        self.data_config = parse_data_config(data_config_path)
        self.class_path = self.data_config["names"]
        self.classes = load_classes(self.class_path)

        self.train_path = self.data_config["train"]
        self.valid_path = self.data_config["valid"]
        self.image_folder = self.valid_path

        self.cuda = torch.cuda.is_available() and use_cuda

        self.image_dir = "data/{}/image".format(task)
        self.voc_dir = "data/{}/voc".format(task)
        self.sample_dir = "data/{}/sample".format(task)
        self.input_mp4_path = "data/{}/input.mp4".format(task)
        self.output_mp4_path = "data/{}/output.mp4".format(task)


# default configurations for current task
task = "street"
data_config_path = "config/{}.data".format(task)
class_path = "data/{}/{}.names".format(task, task)
task_yolo_config = YOLOConfig(
    data_config_path=data_config_path
)
