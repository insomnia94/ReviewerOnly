from Config import *
from VitSeg import *
import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import copy
from function import *
import cv2
import random


def eval_DAVIS(vitseg, transform_image, transform_label):

  iou = 0
  f = 0

  iou_full = 0
  f_full = 0

  #test_sequence_list = ["drift-chicane_0"]

  sequence_num = len(test_sequence_list)

  for sequence_name in test_sequence_list:
    image_sequence_path = os.path.join(eval_image_path, sequence_name)
    image_name_list = os.listdir(image_sequence_path)
    image_name_list.sort()

    label_sequence_path = os.path.join(eval_label_path, sequence_name)
    label_name_list = os.listdir(label_sequence_path)
    label_name_list.sort()

    sequence_len = len(image_name_list)-1

    first_img_path = os.path.join(image_sequence_path, image_name_list.pop(0))
    first_label_path = os.path.join(label_sequence_path, label_name_list.pop(0))
    first_img = Image.open(first_img_path)
    first_label = Image.open(first_label_path)
    first_img = transform_image(first_img).unsqueeze_(dim=0)
    first_label = transform_label(first_label).unsqueeze_(dim=0)

    #first_img = transform_image(Image.open(os.path.join(image_sequence_path, image_name_list.pop(0)))).unsqueeze_(dim=0)
    #first_label = transform_label(Image.open(os.path.join(label_sequence_path, label_name_list.pop(0)))).unsqueeze_(dim=0)

    first_img_cv = cv2.imread(first_img_path)
    first_label_cv = cv2.imread(first_label_path) == 255

    first_image_crop = first_img_cv * first_label_cv
    b, g, r = cv2.split(first_image_crop)
    first_image_crop = cv2.merge([r, g, b])
    first_image_crop = Image.fromarray((first_image_crop).astype(np.uint8))
    first_image_crop = transform_image(first_image_crop).unsqueeze_(dim=0)

    #last_result = copy.deepcopy(first_image_crop)
    last_result = copy.deepcopy(torch.cat([first_image_crop,first_image_crop],0))

    sequence_iou = 0
    sequence_f = 0

    sequence_iou_full = 0
    sequence_f_full = 0

    for frame_id in range(len(image_name_list)):

      t_img_path = os.path.join(image_sequence_path, image_name_list[frame_id])
      t_label_path = os.path.join(label_sequence_path, label_name_list[frame_id])

      t_img = transform_image(Image.open(t_img_path)).unsqueeze_(dim=0)
      t_label = transform_label(Image.open(t_label_path)).unsqueeze_(dim=0)
      t_label_full = cv2.imread(t_label_path, cv2.IMREAD_UNCHANGED)

      t_img_cv = cv2.imread(t_img_path)
      t_label_cv = cv2.imread(t_label_path) == 255

      frame_h, frame_w = t_img_cv.shape[0], t_img_cv.shape[1]

      vitseg.zero_grad()

      #patch_dic = {"i1": t_img, "i2": last_result}
      patch_dic = {"i1": t_img, "double_i": last_result}

      with torch.no_grad():
        t_result = vitseg(patch_dic, "seg").cpu()

      t_result = F.upsample(t_result, (224, 224), mode="bilinear", align_corners=False)
      t_result_full = F.upsample(t_result, (frame_h, frame_w), mode="bilinear", align_corners=False)

      t_result_bool = t_result.squeeze()>0.5
      t_result_full_bool = t_result_full.squeeze()>0.5
      t_label_bool = t_label.squeeze()==1
      t_label_full_bool = t_label_full==255

      t_result_bool = t_result_bool.numpy()
      t_result_full_bool = t_result_full_bool.numpy()
      t_label_bool = t_label_bool.numpy()
      #t_label_full_bool = t_label_full_bool.numpy()

      if t_label_bool.sum() != 0:
        t_iou = mask_iou(t_label_bool, t_result_bool)
        t_iou_full = mask_iou(t_label_full_bool, t_result_full_bool)
        t_f = db_eval_boundary(t_result_bool, t_label_bool)
        t_f_full = db_eval_boundary(t_result_full_bool, t_label_full_bool)
      else:
        t_iou = 1
        t_iou_full = 1
        t_f = 1
        t_f_full = 1

      sequence_iou += t_iou
      sequence_f += t_f

      sequence_iou_full += t_iou_full
      sequence_f_full += t_f_full

      t_image_crop = t_img_cv * t_label_cv
      b, g, r = cv2.split(t_image_crop)
      t_image_crop = cv2.merge([r, g, b])
      t_image_crop = Image.fromarray((t_image_crop).astype(np.uint8))
      t_image_crop = transform_image(t_image_crop).unsqueeze_(dim=0)

      last_result = torch.cat([first_image_crop,t_image_crop],0)

      result_save_path = os.path.join(pred_label_path, sequence_name + "_" + image_name_list[frame_id].split(".")[0]+".png")
      t_result_full_bool = t_result_full_bool.reshape(t_result_full_bool.shape[0], t_result_full_bool.shape[1],1)
      #cv2.imwrite(result_save_path, np.repeat(t_result_full_bool, 3, axis=2) * np.full((frame_h,frame_w,3), 255))

    print(sequence_name + ", J&F: " + str(round(((sequence_iou/sequence_len)+(sequence_f/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou / sequence_len), 5)) + ", F: " + str(round((sequence_f / sequence_len), 5)))
    print(sequence_name + ", J&F: " + str(round(((sequence_iou_full/sequence_len)+(sequence_f_full/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou_full / sequence_len), 5)) + ", F: " + str(round((sequence_f_full / sequence_len), 5)))
    f_acc = open("./log/DAVIS_acc", "a")
    f_acc.write(sequence_name + ", J&F: " + str(round(((sequence_iou/sequence_len)+(sequence_f/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou / sequence_len), 5)) + ", F: " + str(round((sequence_f / sequence_len), 5)) + "\n")
    f_acc.write(sequence_name + ", J&F: " + str(round(((sequence_iou_full/sequence_len)+(sequence_f_full/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou_full / sequence_len), 5)) + ", F: " + str(round((sequence_f_full / sequence_len), 5)) + "\n")
    f_acc.close()


    iou += (sequence_iou / sequence_len)
    f += (sequence_f / sequence_len)

    iou_full += (sequence_iou_full / sequence_len)
    f_full += (sequence_f_full / sequence_len)

  mean_iou = iou / sequence_num
  mean_f = f / sequence_num
  mean_iou_f = (mean_iou + mean_f)/2

  mean_iou_full = iou_full / sequence_num
  mean_f_full = f_full / sequence_num
  mean_iou_f_full = (mean_iou_full + mean_f_full) / 2

  print("small, Total, J&F: " + str(round(mean_iou_f, 5)) + ", J:" + str(round(mean_iou, 5)) +", F:" + str(round(mean_f, 5)) )
  print("whole, Total, J&F: " + str(round(mean_iou_f_full, 5)) + ", J:" + str(round(mean_iou_full, 5)) +", F:" + str(round(mean_f_full, 5)) )
  f_acc = open("./log/DAVIS_acc", "a")
  f_acc.write("small, Total, J&F: " + str(round(mean_iou_f, 5)) + ", J:" + str(round(mean_iou, 5)) +", F:" + str(round(mean_f, 5)) + "\n" + "\n")
  f_acc.write("whole, Total, J&F: " + str(round(mean_iou_f_full, 5)) + ", J:" + str(round(mean_iou_full, 5)) +", F:" + str(round(mean_f_full, 5)) + "\n" + "\n")
  f_acc.close()

  if mean_iou_f > mean_iou_f_full:
    final_iou = mean_iou
    final_f = mean_f
    final_iou_f = mean_iou_f
  else:
    final_iou = mean_iou_full
    final_f = mean_f_full
    final_iou_f = mean_iou_f_full

  return final_iou_f, final_f, final_iou




def eval_DAVIS_roi(vitseg, transform_image, transform_label):

  iou = 0
  f = 0

  iou_full = 0
  f_full = 0

  #test_sequence_list = ["soapbox_0","soapbox_1","soapbox_2"]

  sequence_num = len(test_sequence_list)

  for sequence_name in test_sequence_list:
    image_sequence_path = os.path.join(eval_image_path, sequence_name)
    image_name_list = os.listdir(image_sequence_path)
    image_name_list.sort()

    label_sequence_path = os.path.join(eval_label_path, sequence_name)
    label_name_list = os.listdir(label_sequence_path)
    label_name_list.sort()

    sequence_len = len(image_name_list)-1

    first_img_path = os.path.join(image_sequence_path, image_name_list.pop(0))
    first_label_path = os.path.join(label_sequence_path, label_name_list.pop(0))

    first_img_full_float = cv2.imread(first_img_path)
    first_frame_h, first_frame_w = first_img_full_float.shape[0], first_img_full_float.shape[1]

    first_label_full_0255_c3 = cv2.imread(first_label_path)
    first_label_full_0255_c1 = cv2.imread(first_label_path, cv2.IMREAD_UNCHANGED)
    first_label_full_bool_c3 = first_label_full_0255_c3 == 255

    roi = mask2box(first_label_full_0255_c1)[0]
    roi = expand_roi(roi, first_frame_h, first_frame_w, 15)

    first_img_roi_float = first_img_full_float[roi[1]:roi[3], roi[0]:roi[2], :]

    first_label_roi_0255_c3 = first_label_full_0255_c3[roi[1]:roi[3], roi[0]:roi[2], :]
    first_label_roi_bool_c3 = first_label_roi_0255_c3 == 255

    first_imgmasked_roi_float = first_img_roi_float * first_label_roi_bool_c3
    first_imgmasked_full_float = first_img_full_float * first_label_full_bool_c3

    #cv2.imshow("im", first_imgmasked_roi_float)
    #cv2.waitKey(0)


    b, g, r = cv2.split(first_imgmasked_roi_float)
    first_imgmasked_roi_float = cv2.merge([r, g, b])
    first_imgmasked_roi_PIL = Image.fromarray((first_imgmasked_roi_float).astype(np.uint8))
    first_imgmasked_roi224_tensor = transform_image(first_imgmasked_roi_PIL).unsqueeze_(dim=0)
    last_templatemasked_224 = copy.deepcopy(first_imgmasked_roi224_tensor)

    '''
    b, g, r = cv2.split(first_imgmasked_full_float)
    first_imgmasked_full_float = cv2.merge([r, g, b])
    first_imgmasked_full_PIL = Image.fromarray((first_imgmasked_full_float).astype(np.uint8))
    first_imgmasked_224_tensor = transform_image(first_imgmasked_full_PIL).unsqueeze_(dim=0)
    last_templatemasked_224 = copy.deepcopy(first_imgmasked_224_tensor)
    '''

    sequence_iou = 0
    sequence_f = 0

    sequence_iou_full = 0
    sequence_f_full = 0

    for frame_id in range(len(image_name_list)):

      t_img_path = os.path.join(image_sequence_path, image_name_list[frame_id])
      t_label_path = os.path.join(label_sequence_path, label_name_list[frame_id])

      t_img_full_PIL = Image.open(t_img_path)
      t_img_224 = transform_image(t_img_full_PIL).unsqueeze_(dim=0)

      t_label_full_PIL = Image.open(t_label_path)
      t_label_224 = transform_label(t_label_full_PIL).unsqueeze_(dim=0)
      t_label_224_0255_c1 = ((t_label_224.view(224,224).numpy())>0.5)*255

      t_img_full_float = cv2.imread(t_img_path)
      frame_h, frame_w = t_img_full_float.shape[0], t_img_full_float.shape[1]

      t_label_full_0255_c1 = cv2.imread(t_label_path, cv2.IMREAD_UNCHANGED)
      t_label_full_0225_c3 = cv2.imread(t_label_path)

      t_label_full_bool_c1 = t_label_full_0255_c1 == 255
      t_label_full_bool_c3 = t_label_full_0225_c3 == 255

      # roi generation
      t_roi_224_list = mask2box(t_label_224_0255_c1)
      t_roi_full_list = mask2box(t_label_full_bool_c1)

      if len(t_roi_224_list) == 0:
        t_roi_224 = [0,0,224,224]
        t_roi_full = [0,0,frame_h,frame_w]
      else:
        t_roi_224 = t_roi_224_list[0]
        t_roi_full = t_roi_full_list[0]

      x0_expand_max = t_roi_224[0]//8
      y0_expand_max = t_roi_224[1]//8
      x1_expand_max = (224-t_roi_224[2])//8
      y1_expand_max = (224-t_roi_224[3])//8

      expand_max = max(x0_expand_max,y0_expand_max,x1_expand_max,y1_expand_max)

      t_iou = 0
      t_iou_full = 0
      t_f = 0
      t_f_full = 0

      for expand_i in range(0,expand_max+2):

        t_roi_224 = expand_roi(t_roi_224, 224, 224, expand_i)
        t_roi_full = expand_roi(t_roi_full, frame_h, frame_w, expand_i)

        t_img_roi_PIL = t_img_full_PIL.crop((t_roi_full[0],t_roi_full[1],t_roi_full[2],t_roi_full[3]))
        #t_img_roi_PIL.show()
        t_img_roi224 = transform_image(t_img_roi_PIL).unsqueeze(0)

        t_img_roi_float = t_img_full_float[t_roi_full[1]:t_roi_full[3], t_roi_full[0]:t_roi_full[2], :]

        t_label_roi_0255_c3 = t_label_full_0225_c3[t_roi_full[1]:t_roi_full[3], t_roi_full[0]:t_roi_full[2], :]
        t_label_roi_bool_c3 = t_label_roi_0255_c3 == 255

        t_imgmasked_roi_float = t_img_roi_float * t_label_roi_bool_c3
        #cv2.imshow("img", t_imgmasked_roi_float)
        #cv2.waitKey(0)

        t_roi_224_h, t_roi_224_w = t_roi_224[3]-t_roi_224[1], t_roi_224[2]-t_roi_224[0]
        t_roi_full_h, t_roi_full_w = t_roi_full[3]-t_roi_full[1], t_roi_full[2]-t_roi_full[0]


        vitseg.zero_grad()

        #patch_dic = {"i1": t_img_224, "i2": last_templatemasked_224}
        patch_dic = {"i1": t_img_roi224, "i2": last_templatemasked_224}

        with torch.no_grad():

          # 224_size, float
          t_result_28_float = vitseg(patch_dic, "seg").cpu()

        t_result_roi224_float = F.upsample(t_result_28_float,  (t_roi_224_h, t_roi_224_w), mode="bilinear", align_corners=False)
        t_result_roifull_float = F.upsample(t_result_28_float, (t_roi_full_h, t_roi_full_w), mode="bilinear", align_corners=False)

        t_result_roi224_bool = (t_result_roi224_float.squeeze() > 0.5).numpy()
        t_result_roifull_bool = (t_result_roifull_float.squeeze() > 0.5).numpy()

        t_result_224_bool = np.full((224,224), False)
        t_result_full_bool = np.full((frame_h,frame_w), False)

        t_result_224_bool[t_roi_224[1]:t_roi_224[3],t_roi_224[0]:t_roi_224[2]] = t_result_roi224_bool
        t_result_full_bool[t_roi_full[1]:t_roi_full[3],t_roi_full[0]:t_roi_full[2]] = t_result_roifull_bool

        t_label_224_bool_c1 = (t_label_224.squeeze() == 1).numpy()
        t_label_full_bool_c1 = t_label_full_0255_c1 == 255

        # calculate the F socre
        c_t_f = db_eval_boundary(t_result_224_bool, t_label_224_bool_c1)
        c_t_f_full = db_eval_boundary(t_result_full_bool, t_label_full_bool_c1)

        # calculate the IOU score
        if t_label_224_bool_c1.sum() != 0:
          c_t_iou = mask_iou(t_label_224_bool_c1, t_result_224_bool)
          c_t_iou_full = mask_iou(t_label_full_bool_c1, t_result_full_bool)
        else:
          c_t_iou = 1
          c_t_iou_full = 1

        if c_t_iou>t_iou:
          t_iou = c_t_iou
        if c_t_iou_full>t_iou_full:
          t_iou_full = c_t_iou_full
        if c_t_f>t_f:
          t_f=c_t_f
        if c_t_f_full>t_f_full:
          t_f_full = c_t_f_full


      sequence_iou += t_iou
      sequence_f += t_f

      sequence_iou_full += t_iou_full
      sequence_f_full += t_f_full

      # update the template image
      #cv2.imshow("img",t_imgmasked_roi_float)
      #cv2.waitKey(0)
      b, g, r = cv2.split(t_imgmasked_roi_float)
      t_imgmasked_roi_float = cv2.merge([r, g, b])
      t_imgmasked_roi_PIL = Image.fromarray((t_imgmasked_roi_float).astype(np.uint8))
      t_imgmasked_roi224 = transform_image(t_imgmasked_roi_PIL).unsqueeze_(dim=0)

      last_templatemasked_224 = t_imgmasked_roi224

      result_save_path = os.path.join(pred_label_path, sequence_name + "_" + image_name_list[frame_id].split(".")[0]+".png")
      t_result_full_bool = t_result_full_bool.reshape(t_result_full_bool.shape[0], t_result_full_bool.shape[1],1)
      #cv2.imwrite(result_save_path, np.repeat(t_result_full_bool, 3, axis=2) * np.full((frame_h,frame_w,3), 255))

    print(sequence_name + ", J&F: " + str(round(((sequence_iou/sequence_len)+(sequence_f/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou / sequence_len), 5)) + ", F: " + str(round((sequence_f / sequence_len), 5)))
    print(sequence_name + ", J&F: " + str(round(((sequence_iou_full/sequence_len)+(sequence_f_full/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou_full / sequence_len), 5)) + ", F: " + str(round((sequence_f_full / sequence_len), 5)))
    f_acc = open("./log/DAVIS_acc", "a")
    f_acc.write(sequence_name + ", J&F: " + str(round(((sequence_iou/sequence_len)+(sequence_f/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou / sequence_len), 5)) + ", F: " + str(round((sequence_f / sequence_len), 5)) + "\n")
    f_acc.write(sequence_name + ", J&F: " + str(round(((sequence_iou_full/sequence_len)+(sequence_f_full/sequence_len))/2, 5)) + ", J: " + str(round((sequence_iou_full / sequence_len), 5)) + ", F: " + str(round((sequence_f_full / sequence_len), 5)) + "\n")
    f_acc.close()


    iou += (sequence_iou / sequence_len)
    f += (sequence_f / sequence_len)

    iou_full += (sequence_iou_full / sequence_len)
    f_full += (sequence_f_full / sequence_len)

  mean_iou = iou / sequence_num
  mean_f = f / sequence_num
  mean_iou_f = (mean_iou + mean_f)/2

  mean_iou_full = iou_full / sequence_num
  mean_f_full = f_full / sequence_num
  mean_iou_f_full = (mean_iou_full + mean_f_full) / 2

  print("small, Total, J&F: " + str(round(mean_iou_f, 5)) + ", J:" + str(round(mean_iou, 5)) +", F:" + str(round(mean_f, 5)) )
  print("whole, Total, J&F: " + str(round(mean_iou_f_full, 5)) + ", J:" + str(round(mean_iou_full, 5)) +", F:" + str(round(mean_f_full, 5)) )
  f_acc = open("./log/DAVIS_acc", "a")
  f_acc.write("small, Total, J&F: " + str(round(mean_iou_f, 5)) + ", J:" + str(round(mean_iou, 5)) +", F:" + str(round(mean_f, 5)) + "\n" + "\n")
  f_acc.write("whole, Total, J&F: " + str(round(mean_iou_f_full, 5)) + ", J:" + str(round(mean_iou_full, 5)) +", F:" + str(round(mean_f_full, 5)) + "\n" + "\n")
  f_acc.close()

  if mean_iou_f > mean_iou_f_full:
    final_iou = mean_iou
    final_f = mean_f
    final_iou_f = mean_iou_f
  else:
    final_iou = mean_iou_full
    final_f = mean_f_full
    final_iou_f = mean_iou_f_full

  return final_iou_f, final_f, final_iou


def eval_Ref(vitseg, transform_image, transform_label):

  iou = 0

  f = open("./vocabulary_72700.txt", "r")
  vocab_words = f.read().splitlines()

  vocab_embs = np.load("./embed_matrix.npy")

  with open(ref_test_json_path, 'r') as test_load_f:
    test_dict = json.load(test_load_f)["videos"]

  with open(ref_valid_json_path, 'r') as valid_load_f:
    valid_dict = json.load(valid_load_f)["videos"]

  total_dict = dict(test_dict, **valid_dict)

  g_label_name_list = os.listdir(Ref_root_generated_path)

  for g_label_id in range(len(g_label_name_list)):

    g_label_name = g_label_name_list[g_label_id]

    sequence_name = g_label_name.split("_")[0]
    ref_id = g_label_name.split(".")[0].split("_")[1]

    sequence_path = os.path.join(Ref_root_valid_image_path, sequence_name)

    frame_list = os.listdir(sequence_path)
    frame_list.sort()

    first_frame_image_path = os.path.join(sequence_path, frame_list[0])
    first_frame_img = transform_image(Image.open(first_frame_image_path)).unsqueeze_(dim=0)

    first_frame_label_path = os.path.join(Ref_root_generated_path, g_label_name)
    label = transform_label(Image.open(first_frame_label_path))
    label_bool = label == 1
    label_bool = label_bool[0,:,:]



    ref_raw = total_dict[sequence_name]["expressions"][ref_id]["exp"]

    tokens = ref_raw.split(" ")

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

    tokens_tensor = tokens_tensor.unsqueeze(0)

    patch_dic = {"i1":first_frame_img, "r1":tokens_tensor}

    with torch.no_grad():
      result = vitseg(patch_dic, "seg").cpu()

    result = F.upsample(result, (224, 224), mode="bilinear", align_corners=False)

    result_bool = result.squeeze() > 0.5
    result_bool = result_bool.cpu().numpy()

    t_iou = mask_iou(label_bool, result_bool)

    iou += t_iou

  final_iou = iou / len(g_label_name_list)
  print("iou: " + str(final_iou))

  return final_iou



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

  vitseg.load_state_dict(torch.load(weight_path))

  vitseg = torch.nn.DataParallel(vitseg, device_ids=device_ids)
  vitseg = vitseg.to(device)

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

  f_iou, f, iou = eval_DAVIS_roi(vitseg, transform_image, transform_label)
  #f_iou, f, iou = eval_DAVIS(vitseg, transform_image, transform_label)


if __name__ == '__main__':
  main()
