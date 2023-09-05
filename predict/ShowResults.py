from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os

imagepath = 'data/230801/test_230801' #需要加载的测试图片的文件路径
savepath = 'data/230801/construction_230801' #保存测试图片的路径
config_file = 'work_dirs/230803/construction/yolov3_d53_mstrain-2688_1512_1class_construction.py' #网络模型
checkpoint_file = 'work_dirs/230803/construction/best_bbox_mAP_epoch_255.pth'  #训练好的模型参数
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

for filename in os.listdir(imagepath):
    img = os.path.join(imagepath, filename)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)
    show_result_pyplot(model, img, result, score_thr=0.5, out_file=out_file) # palette=[(50,205,50),(220,20,60)],