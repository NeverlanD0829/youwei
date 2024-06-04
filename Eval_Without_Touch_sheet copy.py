import numpy as np
import cv2
import xlwt
import xlrd
import datetime
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from nets import get_network
from nets.MobileNetV2 import MobileNetV2
from dataset import MyDataset
# from nets.MobileNet_torch import *
from torchvision import transforms


test_pic_dir = "/home/chen/Desktop/data/Dataset_D/"

def main():
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    check_path = '/home/chen/Desktop/data/train_data/2024-06-01-17-37-MobileNetV2/fold_3_epoch_16_test_loss_3.023e-06.pth'
    model = MobileNetV2()
    model.load_state_dict(torch.load(check_path),strict=False)
    model.to(device)
    model.eval()

    # Initialize real_data (replace with your initialization method)
    real_weight =[]
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量，并且将值缩放到[0, 1]
    ])

    for sample_num in tqdm(range(1,56),desc='Saving Data',unit='File'):
        dirs = os.listdir(test_pic_dir + str(sample_num))
        dirs = sorted(dirs)
        num = 1
        ws = wb.add_sheet(str(sample_num))
        ws.write(0, 0, '序号')
        ws.write(0, 1, '预测值')
        ws.write(0, 2, '真实值')
        ws.write(0, 3, '误差')
        ws.write(0, 4, 'FPS(推理时间/ms)')

        pre_weight_sum = 0.0  # 用于累积预测值的和
        real_weight_sum = 0.0  # 用于累积真实值的和
        fps_sum = 0.0  # 用于累积真实值的和

        labels_csv_file = 'data_label.csv'
        try:
            with open(labels_csv_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                labels_dict = {int(row[0]): float(row[1]) for row in csv_reader}
                label_index = sample_num
                label_value = labels_dict.get(label_index)
                real_weight.append(label_value)
        except FileNotFoundError:
            print(f"Error: File '{labels_csv_file}' not found.")

        rgb_dir = f"/home/chen/Desktop/data/Dataset_D/{sample_num}/color/"
        depth_dir = f"/home/chen/Desktop/data/Dataset_D/{sample_num}/depth/"
        pic_num = os.listdir(rgb_dir)
        for j in range(1,len(pic_num)+1):
            rgb_pic = f"{rgb_dir}/{j}.png"
            depth_pic = f"{depth_dir}/{j}.png"
            img_rgb = cv2.imread(rgb_pic)
            img_rgb = cv2.resize(img_rgb, (300, 300))
            b, g, r = cv2.split(img_rgb)
            thresh, img_rgb = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
            img_depth = cv2.imread(depth_pic,cv2.IMREAD_GRAYSCALE)
            img_depth = cv2.resize(img_depth, (300, 300))
            concatenated_image = np.dstack((img_rgb,img_depth))

            img_normalized = concatenated_image / 255.0
            img_tensor = transform(img_normalized).float()
            img_tensor = img_tensor.unsqueeze(0).to(device)

        # for pic in dirs:
        #     pic_dir = os.path.join(test_pic_dir, str(sample_num), pic)
        #     img = cv2.imread(pic_dir)
        #     img = cv2.resize(img, (300, 300))
        #     b, g, r = cv2.split(img)
        #     thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
        #     img_normalized = img2 / 255.0
        #     img_tensor = transform(img_normalized).float()
        #     img_tensor = img_tensor.unsqueeze(0).to(device)

            # Measure time for forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            with torch.no_grad():
                a = model(img_tensor)
                b = a[0].cpu().numpy()
                pre_weight = b[0]

            end_time.record()
            torch.cuda.synchronize()  # 确保 GPU 完成了前面的所有操作
            elapsed_time = start_time.elapsed_time(end_time)  # milliseconds
            fps = elapsed_time  # Convert milliseconds to seconds

            pre_weight_sum += pre_weight
            real_weight_sum += real_weight[sample_num - 1]
            fps_sum += fps

            # ws.write(num, 0, int(j.split('.')[0]))
            ws.write(num, 0, j)
            ws.write(num, 1, float(pre_weight))
            ws.write(num, 2, real_weight[sample_num - 1])
            loss = abs(float(pre_weight) - float(real_weight[sample_num - 1]))
            ws.write(num, 3, loss)
            ws.write(num, 4, fps)
            pre_weight = None
            num = num + 1
        # 计算平均值
        average_pre_weight = pre_weight_sum / len(pic_num)
        average_real_weight = real_weight_sum / len(pic_num)
        ws.write(num, 0, '平均值')
        ws.write(num, 1, average_pre_weight)
        ws.write(num, 2, average_real_weight)
        loss = abs(average_pre_weight - average_real_weight)
        ws.write(num, 3, loss)
        ws.write(num, 4, fps_sum/len(pic_num))

        wb.save('单种报告RGBD.xls')
    tqdm.write(f'finish ' + str(sample_num) + '/55')


if __name__ == '__main__':
    main()