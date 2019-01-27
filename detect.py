from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from yolo_config import task_yolo_config

image_folder = task_yolo_config.sample_dir
current_weights_path = "checkpoints/200.weights"

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(task_yolo_config.model_config_path, img_size=task_yolo_config.img_size)
model.load_weights(current_weights_path)

if task_yolo_config.cuda:
    model.cuda()

model.eval()  # Set in evaluation mode
dataloader = DataLoader(ImageFolder(image_folder, img_size=task_yolo_config.img_size),
                        batch_size=task_yolo_config.batch_size, shuffle=False, num_workers=task_yolo_config.n_cpu)

Tensor = torch.cuda.FloatTensor if task_yolo_config.cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, len(task_yolo_config.classes), task_yolo_config.conf_thres,
                                         task_yolo_config.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print('\nSaving images:')
# Iterate through images and save plot of detections
bbox_colors = random.sample(colors, len(task_yolo_config.classes))
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    # img = np.array(Image.open(path))
    img = cv2.imread(path)[..., ::-1]
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (task_yolo_config.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (task_yolo_config.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = task_yolo_config.img_size - pad_y
    unpad_w = task_yolo_config.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        # bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print('\t+ Label: %s, Conf: %.5f' % (task_yolo_config.classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = bbox_colors[int(cls_pred)]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=task_yolo_config.classes[int(cls_pred)], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
