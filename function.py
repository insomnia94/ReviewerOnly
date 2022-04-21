import numpy as np
import torch.nn.functional as F
import cv2
import skimage



# convert the probability map (h,w), 0~1, to the binary mask (h,w), 0 or 255
def prob2binary(prob_map):
	height, width = prob_map.shape[:2]
	mask_binary = prob_map > 0.5
	full = np.full((height, width), 255)
	mask_binary = mask_binary * full
	mask_binary = mask_binary.astype(np.uint8)
	return mask_binary

# resize the output of the Foreground Segmentaion network [1,1,244,244], and return the final binary (0 or 255) mask [h,w], and probability (0~1) map [h,w]
def pred_resize(output_mask, target_h, target_w):
	mask_resize_prob = F.upsample(output_mask, (target_h, target_w), mode="bilinear", align_corners=False)
	mask_resize_prob = mask_resize_prob.detach().cpu().numpy()
	mask_resize_prob = np.reshape(mask_resize_prob, (target_h, target_w))
	mask_resize_binary = prob2binary(mask_resize_prob)

	return mask_resize_binary, mask_resize_prob

# calculate the IOU between two segmentation masks
def mask_iou(labels_bool, preds_bool):
  #labels_bool = (labels == 255)
  #pred_bool = (predictions == 255)

  labels_bool_sum = (labels_bool).sum()
  pred_bool_sum = (preds_bool).sum()

  if (labels_bool_sum > 0) or (pred_bool_sum > 0):
    intersect_all = np.logical_and(labels_bool, preds_bool)
    intersect = intersect_all.sum()
    union = labels_bool_sum + pred_bool_sum - intersect

  return float(intersect) / float(union)




def seg2bmap(seg,width=None,height=None):
	seg = seg.astype(np.bool)
	seg[seg > 0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width = seg.shape[1] if width is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e = np.zeros_like(seg)
	s = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:, :-1] = seg[:, 1:]
	s[:-1, :] = seg[1:, :]
	se[:-1, :-1] = seg[1:, 1:]

	b = seg^e | seg^s | seg^se
	b[-1,:] = seg[-1,:]^e[-1,:]
	b[:,-1] = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+np.floor((y-1)+height / h)
					i = 1+np.floor((x-1)+width / h)
					bmap[j,i] = 1;
	return bmap



def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask);
	gt_boundary = seg2bmap(gt_mask);

	from skimage.morphology import binary_dilation,disk

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg = np.sum(fg_boundary)
	n_gt = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall);

	return F

def generate_ROI(box_list, original_w, original_h, ratio):

	top_x_list = []
	top_y_list = []
	bottom_x_list = []
	bottom_y_list = []

	for i in range(len(box_list)):
		top_x_list.append(box_list[i][0])  # x1
		top_y_list.append(box_list[i][1])  # y1
		bottom_x_list.append(box_list[i][2])  # x2
		bottom_y_list.append(box_list[i][3])  # y2

	# the bounding box of all bounding boxes of ground-truth
	top_x = min(top_x_list)
	top_y = min(top_y_list)
	bottom_x = max(bottom_x_list)
	bottom_y = max(bottom_y_list)

	w = bottom_x - top_x
	h = bottom_y - top_y

	top_x = int(top_x - ratio * w)
	top_y = int(top_y - ratio * h)
	bottom_x = int(bottom_x + ratio * w)
	bottom_y = int(bottom_y + ratio * w)

	if top_x < 0:
		top_x = 0
	if top_y < 0:
		top_y = 0
	if bottom_x > original_w:
		bottom_x = original_w
	if bottom_y > original_h:
		bottom_y = original_h

	return top_x, top_y, bottom_x, bottom_y

def mask2box(mask):
	# a list to store the x,y,w,h for each bounding box
	box_list = []

	mask_shape = mask.shape
	mask_h = mask_shape[0]
	mask_w = mask_shape[1]

	# all possible values referring to different objects
	possible_values = np.unique(mask)

	# remove the first value cos it is 0
	possible_values = possible_values[1:]

	# the number of all possible values
	num_values = len(possible_values)

	for i in range(num_values):
		# True or False matrix
		bool_matrix = mask == possible_values[i]

		# 0 or 255 matrix
		binary_matrix = np.zeros((mask_h, mask_w)) + np.full((mask_h, mask_w), 255) * bool_matrix
		binary_matrix = binary_matrix.astype(np.uint8)

		contours, _ = cv2.findContours(binary_matrix, 3, 2)

		contour_box_list = []

		for contour_id in range(len(contours)):
			x, y, w, h = cv2.boundingRect(contours[contour_id])
			contour_box_list.append([x, y, x+w, y+h])

		margin_rate = 0

		top_x, top_y, bottom_x, bottom_y = generate_ROI(contour_box_list, mask_w, mask_h, margin_rate)

		box_list.append([top_x, top_y, bottom_x, bottom_y])

	return box_list

def expand_roi(roi, h, w, expand_ratio):

	h_patch = h // 28
	w_patch = w // 28

	x0, y0, x1, y1 = (roi[0]//w_patch)*w_patch-w_patch*expand_ratio, (roi[1]//h_patch)*h_patch-h_patch*expand_ratio, (roi[2]//w_patch+1)*w_patch+w_patch*expand_ratio, (roi[3]//h_patch+1)*h_patch+h_patch*expand_ratio

	if x0 < 0:
		x0 = 0
	if y0 < 0:
		y0 = 0
	if x1 > w:
		x1 = w
	if y1 > h:
		y1 = h

	roi = [x0,y0,x1,y1]

	return roi




