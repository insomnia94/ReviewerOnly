# this python script is used to test a simple image using 3C foreground segmentation network

import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import os
from datetime import datetime
import torch.nn.functional as F
import cv2

from Config_3C import *
from VitSeg import *
from function import *

def main():
  USE_CUDA = torch.cuda.is_available()
  device = torch.device("cuda:0" if USE_CUDA else "cpu")

  vitseg = ViTSeg(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout)

  vitseg.load_state_dict(torch.load(weight_path))
  vitseg.eval()
  vitseg = vitseg.cuda()

  input_path = "./test_images/test1.jpg"

  transform_image = transforms.Compose([
    transforms.Resize((trans_resize_size, trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  image = Image.open(input_path)

  w = image._size[0]
  h = image._size[1]

  image = transform_image(image)

  image = Variable(image.to(device))
  image.unsqueeze_(dim=0)

  patch_dic = {"i1": image}

  mask = vitseg(patch_dic, "seg")

  mask_binary, _ = pred_resize(mask, h, w)

  cv2.imshow("1", mask_binary)
  cv2.waitKey(0)


if __name__ == '__main__':
  main()