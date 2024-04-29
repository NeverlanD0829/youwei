import shutil                                                            #  复制、移动、重命名和删除文件或目录
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading                                                        #Python 中用于支持多线程编程的标准库模块                                                   #解析命令行参数
import os
import sys                                                              #针对于python解释器相关的变量和方法
from pathlib import Path                                               # 提供了一种更面向对象的方式来处理文件系统路径和文件。它引入了 Path 类，可以轻松进行路径操作而无需使用字符串连接。
import cv2
import torch
import torch.backends.cudnn as cudnn
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, time_sync
from main_window import Ui_MainWindow
from datetime import datetime
from nets.MobileNetV2 import *
from Predict_Without_Touch_single import *


FILE = Path(__file__).resolve()                                        # 获取当前脚本文件的路径（绝对路径）  E:\Desktop\code\main_UI.py
ROOT = FILE.parents[0]                                                 # 获取当前脚本文件的父目录（上级目录） E:\Desktop\code\main_UI.py
# 检查 ROOT 和当前工作目录是否在同一驱动器上
if os.path.splitdrive(ROOT)[0].lower() == os.path.splitdrive(os.getcwd())[0].lower():
    # 如果它们在同一驱动器上，计算相对路径
    ROOT = Path(os.path.relpath(ROOT, os.getcwd()))                    # 计算 ROOT 相对于当前工作目录的相对路径。这通常用于将一个路径转换为相对于另一个路径的相对路径。
else:
    # 如果它们在不同的驱动器上，将 ROOT 设置为绝对路径
    ROOT = ROOT.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                                         # 将 ROOT（YOLOv5 项目的根目录）添加到 Python 的模块搜索路径 sys.path 中

# 添加一个关于界面
# 窗口主类
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)                                              #self参数传递给 setupUi，将用户界面的元素添加到当前主窗口实例中。

        # 信号槽
        self.btn_detect_img.clicked.connect(self.detect_img)
        self.btn_load_img.clicked.connect(self.load_img)
        self.btn_detect_cam.clicked.connect(self.open_cam)
        self.btn_detect_video.clicked.connect(self.open_mp4)
        self.action_changemodel.triggered.connect(self.choosemodel)
        self.btn_video_stop.clicked.connect(self.close_vid)             #视频检测页面添加 停止检测按钮
        self.btn_video_stop.setEnabled(False)                           #关闭后禁用

        # 图片显示的大小，初始宽500，高500
        self.width = 300
        self.height = 300
        pix_rgb = QPixmap("images/UI/img.png")
        pix_rgb.scaled(self.width, self.height, Qt.KeepAspectRatio)         # 使用 scaled 方法对图像进行  按比例 缩放
        self.label_img.setPixmap(pix_rgb)


        pix_d = QPixmap("images/UI/img_d.png")
        pix_d.scaled(self.width, self.height, Qt.KeepAspectRatio)  # 使用 scaled 方法对图像进行  按比例 缩放
        self.label_img_d.setPixmap(pix_d)
        self.label_img.setScaledContents(True)                     #设置标签的内容是否按比例进行缩放以适应标签的大小
        self.label_img_d.setScaledContents(True)

        # 视屏区域初始化
        movie = QMovie("images/UI/Dongtu.gif")
        pix_rgb.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.label_left_video.setMovie(movie)
        movie.start()
        self.label_left_video.setScaledContents(True)

        ## TODO 模型相关参数      类的初始化部分,一些变量的初始化和对视频读取线程的设置
        # 图片读取进程
        self.output_size = 300
        self.img2predict = ""
        self.device = 'cuda:0'                                              #在GPU上运行
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        # self.model = self.model_load(weights="model/CNN/2023-12-26-22-20-MobileNetV2/fold_3_epoch_30_test_loss_6.66e-07.pth",device=self.device)                # 默认情况下的权重文件及设备
        self.model = self.model_load()

        # 图片和视频检测前后数据保存,类的初始化部分，用于设置一些与视频处理相关的属性。
        self.output_folder = 'output/'
        self.vid_writer = None                          #初始化了一个属性 vid_writer，用于表示视频写入器。在这里，初始值为 None，可能会在后续的代码中被赋予一个实际的视频写入器对象
        self.cap = cv2.VideoCapture()
        self.cap_writer = None

    @torch.no_grad()                                                        #装饰器，装饰的函数或代码块中的所有运算设置为不计算梯度
    def model_load(self, weights="", device=''):
        model = MobileNetV2()
        model.load_state_dict(torch.load('model/CNN/2023-12-26-22-20-MobileNetV2/fold_3_epoch_30_test_loss_6.66e-07.pth'), strict=False)
        print("深度学习模型加载完成!")
        return model

    def choosemodel(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择训练好的权重文件文件', '.', '*.pth;*.pt')
        if not fileName:
            QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QMessageBox.Ok,defaultButton=QMessageBox.Ok)
        else:
            print('加载weights文件地址为：' + str(fileName))
            self.model = self.model_load(weights=os.path.relpath(fileName), device=self.device)  # todo 指明模型加载的位置的设备
        return

    def load_img(self):
        # 选择图片文件进行读取
        fileName_rgb, _ = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        fileName_d, _ = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        # file_name_prefix = os.path.splitext(os.path.basename(os.path.dirname(fileName_rgb)))[0]                           #RGB图片所在文件夹前缀名
        file_name_prefix = os.path.splitext(os.path.basename(os.path.dirname(os.path.dirname(fileName_rgb))))[0]            ##RGB图片所在文件夹上一文件夹前缀名，不包含扩展
        # print(file_name_prefix)
        real_weight = [0.1448, 0.1388, 0.1501, 0.1726, 0.1572, 0.1183, 0.157, 0.1618, 0.1991, 0.254, 0.1718,
                       0.152, 0.1602, 0.2533, 0.1736, 0.12, 0.2158, 0.1268, 0.2195, 0.1121, 0.1159, 0.2041,
                       0.1257, 0.1702, 0.1569, 0.109, 0.1218, 0.1784, 0.126, 0.1857, 0.1906, 0.2054, 0.2312,
                       0.2024, 0.2649, 0.1791, 0.1798, 0.1661, 0.1875, 0.1743, 0.1839, 0.1726, 0.1887, 0.1634,
                       0.1893, 0.1485, 0.1501, 0.2382, 0.1482, 0.1389, 0.923, 0.1041, 0.1151, 0.1171, 0.2537]
        print(real_weight[1], real_weight[int(file_name_prefix)-1])
        if fileName_rgb and fileName_d == " ":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            image_rgb = QImage(fileName_rgb)
            image_d = QImage(fileName_d)
            pixmap_rgb = QPixmap.fromImage(image_rgb)
            pixmap_d = QPixmap.fromImage(image_d)

            self.label_img.setPixmap(pixmap_rgb)
            self.label_img_d.setPixmap(pixmap_d)
            self.label_left.setText(str(real_weight[int(file_name_prefix)-1]))
            model = self.model
            model.eval()
            try:
                predict_data = predict(fileName_rgb)
                predict_tensor = torch.tensor(predict_data)
                predict_tensor = torch.unsqueeze(predict_tensor, 1)
                # 使用模型进行预测
                with torch.no_grad():
                    prediction = model(predict_tensor)
                    pre = str(prediction.item())[:6]
                    print(pre)
                    self.label_right.setText(pre)

            except:
                print('Open Error! Try again!')
                self.label_right.setText("Predict ERROR!")
        return

    def detect_img(self):

        return
    def open_cam(self):
        self.btn_detect_cam.setEnabled(False)
        self.btn_detect_video.setEnabled(False)
        self.btn_video_stop.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        # camera_num = 0
        # self.cap = cv2.VideoCapture(camera_num)

        th = threading.Thread(target=self.detect_video)
        th.start()
        return
    #
    #
    def open_mp4(self):
    #     fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
    #     flag = self.cap.open(fileName)
    #     if flag:
    #
    #         self.btn_detect_cam.setEnabled(False)
    #         self.btn_detect_video.setEnabled(False)
    #         self.btn_video_stop.setEnabled(True)
    #         self.vid_source = fileName
    #         self.webcam = False
    #         th = threading.Thread(target=self.detect_video)
    #         th.start()
    #         self.btn_video_stop.setEnabled(True)
        return
    #
    def detect_video(self):
    #     # 获取当前系统时间，作为img文件名
    #     now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    #     file_extension = self.vid_source.split('.')[-1]
    #     new_filename = now + '.' + file_extension # 获得文件后缀名
    #     file_path = self.output_folder + 'video_input/' + new_filename
    #
    #     if self.vid_source != '0':
    #         shutil.copy(self.vid_source,file_path)
    #
    #     file_path = self.output_folder + 'video_output/' + new_filename
    #     fps = self.cap.get(cv2.CAP_PROP_FPS)
    #     w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     self.vid_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #
    #     # pass
    #     model = self.model
    #     output_size = self.output_size
    #     # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
    #     imgsz = [640, 640]  # inference size (pixels)
    #     conf_thres = 0.25  # confidence threshold
    #     iou_thres = 0.45  # NMS IOU threshold
    #     max_det = 1000  # maximum detections per image
    #     # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    #     view_img = False  # show results
    #     save_txt = False  # save results to *.txt
    #     save_conf = False  # save confidences in --save-txt labels
    #     save_crop = False  # save cropped prediction boxes
    #     nosave = False  # do not save images/videos
    #     classes = None  # filter by class: --class 0, or --class 0 2 3
    #     agnostic_nms = False  # class-agnostic NMS
    #     augment = False  # ugmented inference
    #     visualize = False  # visualize features
    #     line_thickness = 3  # bounding box thickness (pixels)
    #     hide_labels = False  # hide labels
    #     hide_conf = False  # hide confidences
    #     half = False  # use FP16 half-precision inference
    #     dnn = False  # use OpenCV DNN for ONNX inference
    #     source = str(self.vid_source)
    #     webcam = self.webcam
    #     device = select_device(self.device)
    #     stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    #     imgsz = check_img_size(imgsz, s=stride)  # check image size
    #     save_img = not nosave and not source.endswith('.txt')  # save inference images
    #     # Dataloader
    #     if webcam:
    #         view_img = check_imshow()
    #         cudnn.benchmark = True  # set True to speed up constant image size inference
    #         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    #         bs = len(dataset)  # batch_size
    #     else:
    #         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    #         bs = 1  # batch_size
    #     vid_path, vid_writer = [None] * bs, [None] * bs
    #     # Run inference
    #     if pt and device.type != 'cpu':
    #         model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    #     dt, seen = [0.0, 0.0, 0.0], 0
    #     for path, im, im0s, vid_cap, s in dataset:
    #         t1 = time_sync()
    #         im = torch.from_numpy(im).to(device)
    #         im = im.half() if half else im.float()  # uint8 to fp16/32
    #         im /= 255  # 0 - 255 to 0.0 - 1.0
    #         if len(im.shape) == 3:
    #             im = im[None]  # expand for batch dim
    #         t2 = time_sync()
    #         dt[0] += t2 - t1
    #         # Inference
    #         # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    #         pred = model(im, augment=augment, visualize=visualize)
    #         t3 = time_sync()
    #         dt[1] += t3 - t2
    #         # NMS
    #         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #         dt[2] += time_sync() - t3
    #         # Second-stage classifier (optional)
    #         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    #         # Process predictions
    #         for i, det in enumerate(pred):  # per image
    #             seen += 1
    #             if webcam:  # batch_size >= 1
    #                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
    #                 s += f'{i}: '
    #             else:
    #                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    #             p = Path(p)  # to Path
    #             # save_path = str(save_dir / p.name)  # im.jpg
    #             # txt_path = str(save_dir / 'labels' / p.stem) + (
    #             #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    #             s += '%gx%g ' % im.shape[2:]  # print string
    #             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #             imc = im0.copy() if save_crop else im0  # for save_crop
    #
    #             # 检测结果显示在界面
    #             self.result = cv2.cvtColor(imc, cv2.COLOR_BGR2BGRA)
    #             self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
    #             self.QtImg = QImage(self.result.data, self.result.shape[1], self.result.shape[0], QImage.Format_RGB32)
    #             self.QtImg = self.QtImg.scaled(500,500, Qt.KeepAspectRatio)
    #             self.label_left_video.setPixmap(QPixmap.fromImage(self.QtImg))
    #             self.label_left_video.setScaledContents(True) # 设置图像自适应界面大小
    #
    #
    #             #self.cap_writer.write(imc)
    #
    #
    #
    #
    #             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    #             if len(det):
    #                 # Rescale boxes from img_size to im0 size
    #                 det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    #
    #                 # Print results
    #                 for c in det[:, -1].unique():
    #                     n = (det[:, -1] == c).sum()  # detections per class
    #                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #
    #                 # Write results
    #                 for *xyxy, conf, cls in reversed(det):
    #                     if save_txt:  # Write to file
    #                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
    #                             -1).tolist()  # normalized xywh
    #                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
    #                         # with open(txt_path + '.txt', 'a') as f:
    #                         #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
    #
    #                     if save_img or save_crop or view_img:  # Add bbox to image
    #                         c = int(cls)  # integer class
    #                         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    #                         annotator.box_label(xyxy, label, color=colors(c, True))
    #                         # if save_crop:
    #                         #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
    #                         #                  BGR=True)
    #             # Print time (inference-only)
    #             LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    #             # Stream results
    #             # Save results (image with detections)
    #             im0 = annotator.result()
    #
    #             # 检测结果显示在界面
    #             self.result = cv2.cvtColor(im0, cv2.COLOR_BGR2BGRA)
    #             self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
    #             self.QtImg = QImage(self.result.data, self.result.shape[1], self.result.shape[0], QImage.Format_RGB32)
    #             self.QtImg = self.QtImg.scaled(500,500, Qt.KeepAspectRatio)
    #             self.label_right_video.setPixmap(QPixmap.fromImage(self.QtImg))
    #             self.label_right_video.setScaledContents(True) # 设置图像自适应界面大小
    #
    #             # 保存到视频
    #             if self.vid_source != '0':
    #                 self.vid_writer.write(im0)
    #
    #         if cv2.waitKey(25) & self.stopEvent.is_set() == True:
    #             self.stopEvent.clear()
    #             self.reset_vid()
    #             break
    #     # 检测释放资源
    #     self.reset_vid()
        return
    #
    def reset_vid(self):
    #     self.btn_detect_cam.setEnabled(True)
    #     self.btn_detect_video.setEnabled(True)
    #     self.btn_video_stop.setEnabled(False)
    #     self.cap.release() # 释放video_capture资源
    #     # if self.vid_source!='0':
    #     self.vid_writer.release() # 释放video_writer资源
    #     #self.cap_writer.release() # 摄像头录像
    #
    #     self.vid_source = '0'
    #     self.webcam = True
        return

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    icon = QIcon()
    icon.addPixmap(QPixmap("./images/UI/Without.png"), QIcon.Normal, QIcon.Off)
    mainWindow.setWindowIcon(icon)
    mainWindow.show()
    sys.exit(app.exec_())
