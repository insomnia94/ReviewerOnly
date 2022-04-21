# this is dataset python to generate the one pair of image and label for the training for 3C network

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from PIL import Image
import cv2
import copy

from Config import *
from function import *


class DataSet_cls(Dataset):
  def __init__(self, folder_dataset, transform_image):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image_path"]
    image_name_list = folder_dataset["image_name"]
    label_map = folder_dataset["label"]

    num_samples = len(folder_dataset["image_path"])

    while True:
      sample_id = random.randint(0, num_samples-1)
      image_path = image_path_list[sample_id]
      image_name = str(image_name_list[sample_id]).split("_")[0]
      image = Image.open(image_path)
      try:
        if image.layers == 3:
          break
      except:
        continue

    label = [label_map[image_name]]

    image = self.transform_image(image)
    label = torch.LongTensor(label)

    return image, label

  def __len__(self):
    return len(self.folder_dataset["image_path"])

class DataSet_fore(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image"]
    label_path_list = folder_dataset["label"]

    num_samples = len(folder_dataset["image"])
    sample_id = random.randint(0, num_samples-1)

    image_path = image_path_list[sample_id]
    label_path = label_path_list[sample_id]

    image = Image.open(image_path)
    image = self.transform_image(image)

    label = Image.open(label_path)
    label = self.transform_label(label)

    return image, label

  def __len__(self):
    return len(self.folder_dataset["image"])

class DataSet_DAVIS(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image"]
    label_path_list = folder_dataset["label"]

    num_samples = len(folder_dataset["image"])

    while True:

      sample_id = random.randint(0, num_samples-1)

      image_path = image_path_list[sample_id]
      label_path = label_path_list[sample_id]

      frame_id = int(str(image_path).split("/")[-1].split("_")[2].split(".")[0])
      if frame_id == 0:
        continue

      template_path = label_path_list[sample_id-1]

      image = Image.open(image_path)
      label = Image.open(label_path)
      template = Image.open(template_path)

      image = self.transform_image(image)
      label = self.transform_label(label)
      template = self.transform_label(template)
      break

    return image, label, template

  def __len__(self):
    return len(self.folder_dataset["image"])



class DataSet_YoutubeVOS_double(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset

    target_label_path_list = folder_dataset["target_label_path_list"]
    sequence_name_list = folder_dataset["sequence_name_list"]
    target_id_list = folder_dataset["target_id_list"]
    frame_id_list = folder_dataset["frame_id_list"]
    target_image_path_list = folder_dataset["target_image_path_list"]

    num_target_label = len(target_label_path_list)

    while True:

      sample_id = random.randint(0, num_target_label-1)

      target_label_path = target_label_path_list[sample_id]
      sequence_name = sequence_name_list[sample_id]
      target_id = target_id_list[sample_id]
      frame_id = frame_id_list[sample_id]
      target_image_path = target_image_path_list[sample_id]

      if int(frame_id) == 0:
        continue

      if target_id_list[sample_id] != target_id_list[sample_id-1]:
        continue

      last_target_label_path = target_label_path_list[sample_id-1]
      last_sequence_name = sequence_name_list[sample_id-1]
      last_target_id = target_id_list[sample_id-1]
      last_frame_id = frame_id_list[sample_id-1]
      last_target_image_path = target_image_path_list[sample_id-1]

      target_annotaion_path = os.path.join(You_root_annotation_path, sequence_name)
      annotation_frame_ids = os.listdir(target_annotaion_path)
      annotation_frame_ids.sort()
      first_frame_id = annotation_frame_ids[0].split(".")[0]

      first_target_image_path = You_root_image_path + "/" + sequence_name + "/" + first_frame_id + ".jpg"
      first_target_label_path = You_root_label_path + "/" + sequence_name + "_" + target_id + "_" + first_frame_id + ".png"

      image = Image.open(target_image_path)
      label = Image.open(target_label_path)
      last_image = Image.open(last_target_image_path)
      last_label = Image.open(last_target_label_path)

      last_image_cv = cv2.imread(last_target_image_path)
      last_label_cv = cv2.imread(last_target_label_path) == 255
      last_image_crop = last_image_cv*last_label_cv
      b,g,r = cv2.split(last_image_crop)
      last_image_crop = cv2.merge([r,g,b])
      last_image_crop = Image.fromarray((last_image_crop).astype(np.uint8))

      first_image_cv = cv2.imread(first_target_image_path)
      first_label_cv = cv2.imread(first_target_label_path) == 255
      first_image_crop = last_image_cv*last_label_cv
      #cv2.imshow("img", first_image_crop)
      #cv2.waitKey(0)
      b,g,r = cv2.split(first_image_crop)
      first_image_crop = cv2.merge([r,g,b])
      first_image_crop = Image.fromarray((first_image_crop).astype(np.uint8))

      image = self.transform_image(image)
      last_image = self.transform_image(last_image)
      last_image_crop = self.transform_image(last_image_crop)

      label = self.transform_label(label)
      last_label = self.transform_label(last_label)

      first_image_crop = self.transform_image(first_image_crop)

      combined_template_crop = torch.cat([first_image_crop, last_image_crop], 0)

      break

    return image, last_image, last_image_crop, label, last_label, combined_template_crop

  def __len__(self):
    return len(self.folder_dataset["target_label_path_list"])

class DataSet_YoutubeVOS(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset

    target_label_path_list = folder_dataset["target_label_path_list"]
    sequence_name_list = folder_dataset["sequence_name_list"]
    target_id_list = folder_dataset["target_id_list"]
    frame_id_list = folder_dataset["frame_id_list"]
    target_image_path_list = folder_dataset["target_image_path_list"]

    num_target_label = len(target_label_path_list)

    while True:

      sample_id = random.randint(0, num_target_label-1)

      target_label_path = target_label_path_list[sample_id]
      sequence_name = sequence_name_list[sample_id]
      target_id = target_id_list[sample_id]
      frame_id = frame_id_list[sample_id]
      target_image_path = target_image_path_list[sample_id]

      if int(frame_id) == 0:
        continue

      if target_id_list[sample_id] != target_id_list[sample_id-1]:
        continue

      last_target_label_path = target_label_path_list[sample_id-1]
      last_sequence_name = sequence_name_list[sample_id-1]
      last_target_id = target_id_list[sample_id-1]
      last_frame_id = frame_id_list[sample_id-1]
      last_target_image_path = target_image_path_list[sample_id-1]

      image = Image.open(target_image_path)
      label = Image.open(target_label_path)
      last_image = Image.open(last_target_image_path)
      last_label = Image.open(last_target_label_path)

      last_image_cv = cv2.imread(last_target_image_path)
      last_label_cv = cv2.imread(last_target_label_path) == 255
      last_image_crop = last_image_cv*last_label_cv
      b,g,r = cv2.split(last_image_crop)
      last_image_crop = cv2.merge([r,g,b])
      last_image_crop = Image.fromarray((last_image_crop).astype(np.uint8))

      image = self.transform_image(image)
      last_image = self.transform_image(last_image)
      last_image_crop = self.transform_image(last_image_crop)

      label = self.transform_label(label)
      last_label = self.transform_label(last_label)

      break

    return image, last_image, last_image_crop, label, last_label

  def __len__(self):
    return len(self.folder_dataset["target_label_path_list"])


class DataSet_YoutubeVOS_roi(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset

    target_label_path_list = folder_dataset["target_label_path_list"]
    sequence_name_list = folder_dataset["sequence_name_list"]
    target_id_list = folder_dataset["target_id_list"]
    frame_id_list = folder_dataset["frame_id_list"]
    target_image_path_list = folder_dataset["target_image_path_list"]

    num_target_label = len(target_label_path_list)

    while True:

      sample_id = random.randint(0, num_target_label-1)

      target_label_path = target_label_path_list[sample_id]
      sequence_name = sequence_name_list[sample_id]
      target_id = target_id_list[sample_id]
      frame_id = frame_id_list[sample_id]
      target_image_path = target_image_path_list[sample_id]

      if int(frame_id) == 0:
        continue

      if target_id_list[sample_id] != target_id_list[sample_id-1]:
        continue

      last_target_label_path = target_label_path_list[sample_id-1]
      last_sequence_name = sequence_name_list[sample_id-1]
      last_target_id = target_id_list[sample_id-1]
      last_frame_id = frame_id_list[sample_id-1]
      last_target_image_path = target_image_path_list[sample_id-1]

      image = Image.open(target_image_path)
      label = Image.open(target_label_path)
      last_image = Image.open(last_target_image_path)
      last_label = Image.open(last_target_label_path)

      label_cv_1c = cv2.imread(target_label_path, cv2.IMREAD_UNCHANGED)
      last_label_cv_1c = cv2.imread(last_target_label_path, cv2.IMREAD_UNCHANGED)

      h, w = label_cv_1c.shape[0], label_cv_1c.shape[1]

      expand_ratio = random.randint(3,28)

      current_roi = mask2box(label_cv_1c)[0]
      current_roi = expand_roi(current_roi, h, w, expand_ratio)
      last_roi = mask2box(last_label_cv_1c)[0]
      last_roi = expand_roi(last_roi, h, w, expand_ratio)

      last_label_cv_3c = cv2.imread(last_target_label_path)
      last_label_cv_3c_bool = last_label_cv_3c == 255

      last_image_cv = cv2.imread(last_target_image_path)


      # generate roi

      image = image.crop((current_roi[0],current_roi[1],current_roi[2],current_roi[3]))
      last_image = last_image.crop((last_roi[0],last_roi[1],last_roi[2],last_roi[3]))
      label = label.crop((current_roi[0],current_roi[1],current_roi[2],current_roi[3]))
      last_label = last_label.crop((last_roi[0],last_roi[1],last_roi[2],last_roi[3]))

      last_image_cv_roi = last_image_cv[last_roi[1]:last_roi[3], last_roi[0]:last_roi[2], :]
      last_label_cv_3c_bool_roi = last_label_cv_3c_bool[last_roi[1]:last_roi[3], last_roi[0]:last_roi[2], :]

      last_image_crop = last_image_cv_roi*last_label_cv_3c_bool_roi


      #image.show()
      #last_image.show()
      #label.show()
      #last_label.show()
      #cv2.imshow("img", last_image_crop)
      #cv2.waitKey(0)


      b,g,r = cv2.split(last_image_crop)
      last_image_crop = cv2.merge([r,g,b])
      last_image_crop = Image.fromarray((last_image_crop).astype(np.uint8))

      image = self.transform_image(image)
      last_image = self.transform_image(last_image)
      last_image_crop = self.transform_image(last_image_crop)

      label = self.transform_label(label)
      last_label = self.transform_label(last_label)

      break

    return image, last_image, last_image_crop, label, last_label

  def __len__(self):
    return len(self.folder_dataset["target_label_path_list"])

class DataSet_Ref(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset

    target_label_path_list = folder_dataset["target_label_path_list"]
    sequence_name_list = folder_dataset["sequence_name_list"]
    target_id_list = folder_dataset["target_id_list"]
    frame_id_list = folder_dataset["frame_id_list"]
    target_image_path_list = folder_dataset["target_image_path_list"]
    exp_dict = folder_dataset["exp_dict"]
    vocab_words = folder_dataset["vocab_words"]
    vocab_embs = folder_dataset["vocab_embs"]

    num_target_label = len(target_label_path_list)

    while True:

      sample_id = random.randint(0, num_target_label-1)

      target_label_path = target_label_path_list[sample_id]
      sequence_name = sequence_name_list[sample_id]
      target_id = target_id_list[sample_id]
      frame_id = frame_id_list[sample_id]
      target_image_path = target_image_path_list[sample_id]

      if int(frame_id) == 0:
        continue

      if target_id_list[sample_id] != target_id_list[sample_id-1]:
        continue

      exp_sents = exp_dict[sequence_name]["expressions"]
      exp_sents_copy = copy.deepcopy(exp_sents)
      exp_frames = exp_dict[sequence_name]["frames"]

      for sent_id in exp_sents:
        if exp_sents[sent_id]["obj_id"] != str(target_id):
          exp_sents_copy.pop(sent_id)

      random_sent_id = random.randint(0, len(exp_sents_copy)-1)
      exp_sents_keys = list(exp_sents_copy.keys())
      random_key = exp_sents_keys[random_sent_id]
      raw = exp_sents_copy[random_key]["exp"]

      tokens = raw.split(" ")

      if len(tokens) > query_thre:
        tokens = tokens[0:query_thre]

      while True:
        if len(tokens)<query_thre:
          tokens.append("<pad>")
        else:
          break

      token_embeds = []

      for token in tokens:
        token = token.replace(",","")
        token = token.replace(".","")
        token = token.replace(":","")
        token = token.replace("/","")

        if token in vocab_words:
          word_id = vocab_words.index(token)
        else:
          word_id = 3
        word_emb = vocab_embs[word_id]
        token_embeds.append(word_emb)

      tokens_tensor = torch.FloatTensor(np.array(token_embeds))

      last_target_label_path = target_label_path_list[sample_id-1]
      last_sequence_name = sequence_name_list[sample_id-1]
      last_target_id = target_id_list[sample_id-1]
      last_frame_id = frame_id_list[sample_id-1]
      last_target_image_path = target_image_path_list[sample_id-1]

      image = Image.open(target_image_path)
      label = Image.open(target_label_path)
      last_image = Image.open(last_target_image_path)
      last_label = Image.open(last_target_label_path)

      last_image_cv = cv2.imread(last_target_image_path)
      last_label_cv = cv2.imread(last_target_label_path) == 255
      last_image_crop = last_image_cv*last_label_cv
      b,g,r = cv2.split(last_image_crop)
      last_image_crop = cv2.merge([r,g,b])
      last_image_crop = Image.fromarray((last_image_crop).astype(np.uint8))

      image = self.transform_image(image)
      last_image = self.transform_image(last_image)
      last_image_crop = self.transform_image(last_image_crop)

      label = self.transform_label(label)
      last_label = self.transform_label(last_label)

      break

    return image, last_image, last_image_crop, label, last_label, tokens_tensor

  def __len__(self):
    return len(self.folder_dataset["target_label_path_list"])


