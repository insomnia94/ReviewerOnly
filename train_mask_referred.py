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
import torch.nn.functional as F
import os
from datetime import datetime
from collections import OrderedDict
import random

from Config import *
from VitSeg import *
from DataSet import DataSet_YoutubeVOS
from DataSet import DataSet_YoutubeVOS_roi
from DataSet import DataSet_YoutubeVOS_double
from eval import *

def generate_folder_dataset(root_image_path, root_label_path):

  target_label_name_list = os.listdir(root_label_path)
  target_label_name_list.sort()

  target_label_path_list = []
  sequence_name_list = []
  target_id_list = []
  frame_id_list = []
  target_image_path_list = []

  for target_label_name in target_label_name_list:
    target_label_path = os.path.join(root_label_path, target_label_name)
    sequence_name = target_label_name.split("_")[0]
    target_id = target_label_name.split("_")[1]
    frame_id = target_label_name.split("_")[2].split(".")[0]
    target_image_path = os.path.join(root_image_path, sequence_name, frame_id+".jpg")

    target_label_path_list.append(target_label_path)
    sequence_name_list.append(sequence_name)
    target_id_list.append(target_id)
    frame_id_list.append(frame_id)
    target_image_path_list.append(target_image_path)

  folder_dataset = {"target_label_path_list": target_label_path_list,
                    "sequence_name_list": sequence_name_list,
                    "target_id_list": target_id_list,
                    "frame_id_list": frame_id_list,
                    "target_image_path_list": target_image_path_list}

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

  folder_dataset = generate_folder_dataset(You_root_image_path, You_root_label_path)

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

  #dataset_4C = DataSet_YoutubeVOS_roi(folder_dataset, transform_image, transform_label)
  #dataset_4C = DataSet_YoutubeVOS(folder_dataset, transform_image, transform_label)
  dataset_4C = DataSet_YoutubeVOS_double(folder_dataset, transform_image, transform_label)

  train_dataloader = DataLoader(dataset_4C,
                                shuffle=True,
                                num_workers=worker_num,
                                batch_size=batch_num)

  optimizer = optim.Adam(vitseg.parameters(), lr=lr)

  criterion = nn.BCELoss(size_average=True)

  start_time = datetime.now()

  f_log = open("./log/vos_YoutubeVOS_log", "w")
  f_log.close()

  f_acc = open("./log/DAVIS_acc", "w")
  f_acc.close()

  f_iou_r = 0

  for epoch_id in range(epoch_num):
    for iter_id, data in enumerate(train_dataloader, 0):

      #images, last_images, last_image_crops, labels, last_labels = data
      images, last_images, last_image_crops, labels, last_labels, combined_template_crop = data

      images = Variable(images.to(device))
      labels = Variable(labels.to(device))
      last_images = Variable(last_images.to(device))
      last_labels = Variable(last_labels.to(device))
      last_image_crops = Variable(last_image_crops.to(device))
      combined_template_crop = Variable(combined_template_crop.to(device))

      vitseg.zero_grad()

      #patch_dic = {"i1":images, "i2":last_image_crops}
      patch_dic = {"i1":images, "double_i":combined_template_crop}

      fx = vitseg(patch_dic, "seg")
      fx = F.upsample(fx, scale_factor=8, mode="bilinear")

      ce_loss = criterion(fx.view(-1), labels.view(-1))

      iou_loss = 1 - (fx.min(labels)).sum() / (fx.max(labels).sum())

      bce_loss = F.binary_cross_entropy(fx.view(-1), labels.view(-1), reduce=False)
      focal_loss = (1 * (1 - torch.exp(-bce_loss)) ** 2) * bce_loss
      focal_num = focal_loss.shape[0]
      focal_loss = focal_loss.sum()/focal_num

      smooth = 1
      intersection = (fx.view(-1) * labels.view(-1)).sum()
      dice_loss = (2. * intersection + smooth) / (fx.sum() + labels.sum() + smooth)

      seg_loss = 0*ce_loss + 0*iou_loss + 1*focal_loss + 0*dice_loss

      seg_loss.backward()
      optimizer.step()

      loss_value = seg_loss.detach().cpu().numpy()

      current_time = datetime.now()
      used_time = current_time - start_time

      if iter_id%10 == 0:
        print(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time))
        f_log = open("./log/vos_YoutubeVOS_log", "a")
        f_log.write(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time) + "\n")
        f_log.close()

      if (iter_id%val_iters == 0) and (iter_id>0):
        f_iou, f, iou = eval_DAVIS(vitseg, transform_image, transform_label)
        if f_iou > f_iou_r:
          torch.save(vitseg.module.state_dict(), weight_path)
          print("model saved")
          f_iou_r = f_iou

        print("best: " + str(f_iou_r))
        f_log = open("./log/DAVIS_acc", "a")
        f_log.write("best: " + str(f_iou_r) + "\n")
        f_log.close()

    if (epoch_id % 1 == 0) and (epoch_id > 0):
      f_iou, f, iou = eval_DAVIS(vitseg, transform_image, transform_label)
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