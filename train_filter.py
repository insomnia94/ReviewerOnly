# the input of the network is the original image (3 channels, RGB) and one channel to indicate the segmentation prediction of the last frame

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import os
from datetime import datetime
from collections import OrderedDict
import random

from Config import *
from VitSeg import *
from DataSet import DataSet_DAVIS
from eval import *

def generate_folder_dataset(root_image_path, root_label_path):
  target_frame_name_png_list = os.listdir(root_label_path)
  target_frame_name_png_list.sort()

  image_path_list = []
  label_path_list = []

  for target_frame_name_png in target_frame_name_png_list:
    target_frame_name = target_frame_name_png.split(".")[0]
    target_frame_name_jpg = target_frame_name + ".jpg"

    image_path = os.path.join(root_image_path, target_frame_name_jpg)
    label_path = os.path.join(root_label_path, target_frame_name_png)

    image_path_list.append(image_path)
    label_path_list.append(label_path)

  folder_dataset = {"image": image_path_list, "label": label_path_list}
  return folder_dataset


def main():
  USE_CUDA = torch.cuda.is_available()
  device = torch.device("cuda:0" if USE_CUDA else "cpu")

  torch.manual_seed(1234)
  random.seed(1234)

  vitseg = ViTSeg(
    image_size=image_size,
    patch_size=patch_size,
    num_classes_cls=num_classes_cls,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout)

  if first_train == False:
    vitseg.load_state_dict(torch.load(weight_path))

  vitseg = torch.nn.DataParallel(vitseg, device_ids=device_ids)
  vitseg = vitseg.to(device)

  RL_model = ViTRL(
    image_size=image_size,
    patch_size=patch_size,
    num_classes_cls=num_classes_cls,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout)

  RL_model = torch.nn.DataParallel(RL_model, device_ids=device_ids)
  RL_model= RL_model.to(device)

  folder_dataset = generate_folder_dataset(DAVIS17_root_image_path, DAVIS17_root_label_path)

  transform_image = transforms.Compose([
    transforms.Resize((trans_resize_size, trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  transform_label = transforms.Compose([
    transforms.Resize((trans_resize_size, trans_resize_size)),
    transforms.ToTensor(),
  ])

  dataset_4C = DataSet_DAVIS(folder_dataset, transform_image, transform_label)

  train_dataloader = DataLoader(dataset_4C,
                                shuffle=True,
                                num_workers=worker_num,
                                batch_size=batch_num)

  optimizer = optim.Adam(vitseg.parameters(), lr=lr)

  criterion = nn.BCELoss(size_average=True)
  CE_loss = nn.CrossEntropyLoss(reduction="mean")

  start_time = datetime.now()

  f = open("./log/vos_DAVIS_log", "w")
  f.close()

  f_iou_r = 0

  #f_iou, f, iou = eval_DAVIS_RL(vitseg, RL_model, transform_image, transform_label)

  for epoch_id in range(epoch_num):
    for iter_id, data in enumerate(train_dataloader, 0):

      images, last_images, last_image_crops, labels, last_labels, expand_labels = data

      images = Variable(images.to(device))
      last_images = Variable(last_images.to(device))
      last_image_crops = Variable(last_image_crops.to(device))
      labels = Variable(labels.to(device))
      last_labels = Variable(last_labels.to(device))
      expand_labels = Variable(expand_labels.to(device))


      vitseg.zero_grad()

      patch_dic = {"i1":images, "i2":last_image_crops}

      fx = RL_model(patch_dic, "class")

      #fx = fx.view(-1)
      #y = expand_labels.view(-1)

      RL_loss = CE_loss(fx, expand_labels)
      RL_loss.backward()
      optimizer.step()

      loss_value = RL_loss.detach().cpu().numpy()

      current_time = datetime.now()
      used_time = current_time - start_time

      if iter_id%10 == 0:
        print(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time))
        f = open("./log/vos_DAVIS_log", "a")
        f.write(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time) + "\n")
        f.close()

      #if (iter_id%val_iters == 0) and (iter_id>0):
      if (iter_id%5000 == 0) and (iter_id>0):
        f_iou, f, iou = eval_DAVIS_RL(vitseg, RL_model, transform_image, transform_label)
        if f_iou > f_iou_r:
          torch.save(vitseg.module.state_dict(), weight_path)
          print("model saved")
          f_iou_r = f_iou

        print("best: " + str(f_iou_r))
        f_log = open("./log/DAVIS_acc", "a")
        f_log.write("best: " + str(f_iou_r) + "\n")
        f_log.close()


    f_iou, f, iou = eval_DAVIS_RL(vitseg, RL_model, transform_image, transform_label)

    if f_iou > f_iou_r:
      torch.save(vitseg.module.state_dict(), weight_path)
      print("model saved")
      f_iou_r = f_iou

    print("best: " + str(f_iou_r))
    f_log = open("./log/DAVIS_acc", "a")
    f_log.write("best: " + str(f_iou_r) + "\n")
    f_log.close()

if __name__ == '__main__':
  main()