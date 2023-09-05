import cv2
import os

filepath = "data/230802/video/"
img_fliepath = "frames/"
pathDir = os.listdir (filepath)
for allDir in pathDir:

	video_path = filepath + allDir
	images_path = img_fliepath + allDir.split('.', 1)[0] + "/"
	print(images_path)
	if not os.path.exists(images_path):  # 文件夹不存在则新建
		os.mkdir(images_path)

	cap = cv2.VideoCapture(video_path , cv2.CAP_FFMPEG)
	c = 1
	timeRate = 1  # 截取视频帧的时间间隔

	while(True):
		ret, frame = cap.read()
		FPS = cap.get(5)

		if ret:
			frameRate = int(FPS) * timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
			if(c % frameRate == 0):
				print("开始截取视频第：" + str(c) + " 帧")
				# 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
				cv2.imwrite(images_path + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
			c += 1
			cv2.waitKey(0)
		else:
			print("所有帧都已经保存完成")
			break
	cap.release()