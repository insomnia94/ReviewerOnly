# VIT
image_size = 224
patch_size = 8 # hight and weight of each patch
num_classes_cls = 1000 # category number of final classification network
dim = 1024 # the embedding size of each patch (including multi heads)
depth = 6 # layer num
heads = 16
mlp_dim = 2048 # intermediate dim of feedback module (dim -> mlp_dim -> dim)
dropout = 0.1
emb_dropout = 0.1

# Seg
first_train = True
device_ids = [0]
batch_num = 1
lr = 1e-5
trans_resize_size = 224
worker_num = 1
epoch_num = 1000000000000
val_iters = 1000 # valuate every xxx iterations

query_thre = 15

weight_path = "./weight/model.pkl"

pred_label_path = "./results"

# vos
#DAVIS17_root_image_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_total/image"
#DAVIS17_root_label_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_total/label"
DAVIS17_root_image_path = "/home/smj/DataSet/Saliency/DAVIS2017_intact_test/image"
DAVIS17_root_label_path = "/home/smj/DataSet/Saliency/DAVIS2017_intact_test/label"

You_root_image_path = "/home/smj/DataSet/YoutubeVOS_2019/train/JPEGImages"
You_root_label_path = "/home/smj/DataSet/YoutubeVOS_2019/train/Annotations_flat"
You_root_annotation_path = "/home/smj/DataSet/YoutubeVOS_2019/train/Annotations"

eval_image_path = "/home/smj/DataSet/DAVIS2017_targets/JPEGImages/480p/"
eval_label_path = "/home/smj/DataSet/DAVIS2017_targets/Annotations/480p/"

# Ref VOS
Ref_root_image_path = "/home/smj/DataSet/YoutubeVOS_2019/train/JPEGImages"
Ref_root_label_path = "/home/smj/DataSet/YoutubeVOS_2019/train/Annotations_Ref_flat"

ref_json_path = "/home/smj/DataSet/YoutubeVOS_2021/meta_expressions/train/meta_expressions.json"

# fore
fore_root_image_path = "/home/smj/DataSet/Saliency/COCO2014_MSRA/image"
fore_root_label_path = "/home/smj/DataSet/Saliency/COCO2014_MSRA/label"

# cls
cls_root_image_path = "/home/smj/DataSet/ImageNet_t/train"
cls_root_label_path = "/home/smj/DataSet/ImageNet_t/cls_t_label.txt"


test_sequence_list = ["bike-packing_0",
                      "bike-packing_1",
                      "blackswan_0",
                      "bmx-trees_0",
                      "bmx-trees_1",
                      "breakdance_0",
                      "camel_0",
                      "car-roundabout_0",
                      "car-shadow_0",
                      "cows_0",
                      "dance-twirl_0",
                      "dog_0",
                      "dogs-jump_0",
                      "dogs-jump_1",
                      "dogs-jump_2",
                      "drift-chicane_0",
                      "drift-straight_0",
                      "goat_0",
                      "gold-fish_0",
                      "gold-fish_1",
                      "gold-fish_2",
                      "gold-fish_3",
                      "gold-fish_4",
                      "horsejump-high_0",
                      "horsejump-high_1",
                      "india_0",
                      "india_1",
                      "india_2",
                      "judo_0",
                      "judo_1",
                      "kite-surf_0",
                      "kite-surf_1",
                      "kite-surf_2",
                      "lab-coat_0",
                      "lab-coat_1",
                      "lab-coat_2",
                      "lab-coat_3",
                      "lab-coat_4",
                      "libby_0",
                      "loading_0",
                      "loading_1",
                      "loading_2",
                      "mbike-trick_0",
                      "mbike-trick_1",
                      "motocross-jump_0",
                      "motocross-jump_1",
                      "paragliding-launch_0",
                      "paragliding-launch_1",
                      "paragliding-launch_2",
                      "parkour_0",
                      "pigs_0",
                      "pigs_1",
                      "pigs_2",
                      "scooter-black_0",
                      "scooter-black_1",
                      "shooting_0",
                      "shooting_1",
                      "shooting_2",
                      "soapbox_0",
                      "soapbox_1",
                      "soapbox_2"
                      ]


