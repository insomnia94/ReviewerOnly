# this python script is used to train the 3C network

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import os
from datetime import datetime
from collections import OrderedDict
import random

from Config import *
from DataSet import DataSet_fore
from VitSeg import *

def generate_folder_dataset(root_image_path, root_label_path):
  image_name_list = os.listdir(root_image_path)
  image_name_list.sort()
  image_path_list = []

  for image_name in image_name_list:
    image_path = os.path.join(root_image_path, image_name)
    image_path_list.append(image_path)

  label_name_list = os.listdir(root_label_path)
  label_name_list.sort()
  label_path_list = []

  for label_name in label_name_list:
    label_path = os.path.join(root_label_path, label_name)
    label_path_list.append(label_path)

  folder_dataset = {"image": image_path_list, "label": label_path_list}
  return folder_dataset




def main():

  USE_CUDA = torch.cuda.is_available()
  device = torch.device("cuda:0" if USE_CUDA else "cpu")

  torch.manual_seed(1234)
  random.seed(1234)

  vitseg = ViTSeg(
  image_size = image_size,
  patch_size = patch_size,
  num_classes_cls = num_classes_cls,
  dim = dim,
  depth = depth,
  heads = heads,
  mlp_dim = mlp_dim,
  dropout = dropout,
  emb_dropout = emb_dropout)

  if first_train == False:
      vitseg.load_state_dict(torch.load(weight_path))

  vitseg = torch.nn.DataParallel(vitseg, device_ids=device_ids)
  vitseg = vitseg.to(device)

  folder_dataset = generate_folder_dataset(fore_root_image_path, fore_root_label_path)

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

  dataset_3C = DataSet_fore(folder_dataset, transform_image, transform_label)
  train_dataloader = DataLoader(dataset_3C,
                                shuffle=True,
                                num_workers=worker_num,
                                batch_size=batch_num)
  optimizer = optim.Adam(vitseg.parameters(), lr=lr)
  criterion = nn.BCELoss(size_average=True)
  start_time = datetime.now()

  f = open("./log/fore_log", "w")
  f.close()

  for epoch_id in range(epoch_num):
    for iter_id, data in enumerate(train_dataloader, 0):
      images, labels = data
      images = Variable(images.to(device))
      labels = Variable(labels.to(device))

      #binary_images = images[:,0:1,:,:]

      patch_dic = {"i1":images}
      #patch_dic = {"i1":images, "t1":binary_images}

      vitseg.zero_grad()

      fx = vitseg(patch_dic, 'seg')

      fx = fx.view(-1)
      y = labels.view(-1)
      seg_loss = criterion(fx, y)
      seg_loss.backward()
      optimizer.step()

      loss_value = seg_loss.detach().cpu().numpy()

      current_time = datetime.now()
      used_time = current_time - start_time

      if iter_id%10 == 0:
        print(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time))
        f = open("./log/fore_log", "a")
        f.write(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time) + "\n")
        f.close()

    torch.save(vitseg.module.state_dict(), weight_path)
    print("model saved")


if __name__ == '__main__':
  main()
