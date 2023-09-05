from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import numpy as np
import os
import cv2
import random
import mmcv


config_file = 'work_dirs/230629/construction/yolov3_d53_mstrain-1920_1080_1class.py'
checkpoint_file = 'work_dirs/230629/construction/best_bbox_mAP_epoch_270.pth'#改为自己训练的模型

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
test_img='data/construction_113/construction_new_11/all/construction_138.jpg'#改为自己测试的图片

# test a single image and show the results
#img = 'demo/test.jpg'  # or img = mmcv.imread(img), which will only load it once
img=test_img
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result_138.jpg')
show_result_pyplot(model, img, result)
