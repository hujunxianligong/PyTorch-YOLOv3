from __future__ import division
from models import *
from utils.datasets import *
from utils.parse_config import *
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from yolo_config import task_yolo_config

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get hyper parameters
hyperparams = parse_model_config(task_yolo_config.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(task_yolo_config.model_config_path)
model.load_weights(task_yolo_config.weights_path)
# model.apply(weights_init_normal)

if task_yolo_config.cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(task_yolo_config.train_path), batch_size=task_yolo_config.batch_size, shuffle=False,
    num_workers=task_yolo_config.n_cpu
)

Tensor = torch.cuda.FloatTensor if task_yolo_config.cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(task_yolo_config.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                task_yolo_config.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % task_yolo_config.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (task_yolo_config.checkpoint_dir, epoch))
