from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os
import cv2

score_thr = 0.7 #置信度阈值
time_rate = 1 #截取视频帧的时间间隔

def switch_failure_type(type_id):
    try:
        if type_id == 1:
            type_name ='concrete_weights' # 坠砣
            config_file = 'work_dirs/230803/concrete_weights/yolov3_d53_mstrain-1920_1080_1class_concrete_weights.py' # 网络模型
            checkpoint_file = 'work_dirs/230803/concrete_weights/best_bbox_mAP_epoch_335.pth' # 训练好的模型参数
        elif type_id == 2:
            type_name = 'hanging_string' # 吊弦
            config_file = 'work_dirs/230803/hanging_string/yolov3_d53_mstrain-2688_1512_2class_hanging_string.py' # 网络模型
            checkpoint_file = 'work_dirs/230803/hanging_string/best_bbox_mAP_epoch_325.pth' # 训练好的模型参数           
        return type_name, config_file, checkpoint_file
    
    except Exception as err:
        print(err)


def predict(config_file, checkpoint_file, video_path, image_path, save_path):
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint_file, device=device)

    pathDir = os.listdir (video_path)
    for video in pathDir:
        videoPath = os.path.join(video_path, video)
        cap = cv2.VideoCapture(videoPath , cv2.CAP_FFMPEG)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        c = 1 
        progress = 0.1 
        timeRate = time_rate 

        while(True):
            ret, frame = cap.read()
            FPS = cap.get(5)
            if ret:
                frameRate = int(FPS) * timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
                if(c % frameRate == 0):
                    seconds = int(c/frameRate)
                    # t_m,t_s = divmod(seconds ,60)   
                    # t_h,t_m = divmod(t_m,60)
                    # r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)

                    image_name = video.split('.', 1)[0] + '_' + str(seconds) + 's' + '.jpg'
                    result = inference_detector(model, frame)
                    out_file = os.path.join(save_path, image_name)
                    show_result_pyplot(model, frame, result, score_thr=score_thr, palette='red', out_file=out_file)
                    # print(image_name + '已预测')

                    print('\r', video.split('.', 1)[0] , '进度：', '{:.2%}'.format(c/frame_count), end="")
                    if((c/frame_count) >= progress):
                        progress = progress + 0.1 # 进度每过10%更新一次txt文档
                        with open(image_path + "/test.txt", 'a') as f:
                            f.write(video.split('.', 1)[0] + '\t:\t' + str(c) + '/' + str(frame_count) + '\n')
                c += 1
                cv2.waitKey(0)
            else:
                break
        cap.release() 
        print('\n')
        with open(image_path + "/test.txt", 'a') as f:
            f.write('\n')


def main(types, video_path, image_path):
    for type_id in types:
        type_name, config_file, checkpoint_file = switch_failure_type(type_id)
        save_path = image_path + '/' + type_name
        if not os.path.exists(save_path):  # 文件夹不存在则新建
            os.mkdir(save_path)

        with open(image_path + "/test.txt", 'a') as f:
            f.write(type_name + '\n-----------------------------------------------------------------------\n')
        
        predict(config_file, checkpoint_file, video_path, image_path, save_path)

        with open(image_path + "/test.txt", 'a') as f:
            f.write('\n\n')
    
    print("已完成")
    


if __name__ == '__main__':
    types = [1,2] # 需检测的故障类型的id
    video_path = 'data/video' # 需要加载的测试视频的文件路径
    image_path = 'data/test_result' # 保存测试图片的路径
    main(types, video_path, image_path)
