# import tensorflow as tf
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
from nets.MobileNetV2 import *
from nets import get_network
from nets.AdvanceMobileNetV2 import *
from dataset2 import MyDataset
# from nets.MobileNet_torch import *

test_pic_dir = "./Dataset_D/"


def load_data(self):
    data = []
    labels = []
    labels_csv_file = 'data_label.csv'
    try:
        with open(labels_csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            labels_dict = {int(row[0]): float(row[1]) for row in csv_reader}
    except FileNotFoundError:
        print(f"Error: File '{labels_csv_file}' not found.")
        return [], []

    for i in tqdm(range(1, 51), desc='Load Image', unit='Image'):
        dirs = os.listdir(f"./Dataset_D/{i}/color/")
        for pic_name in dirs:
            rgb_dir = f"./Dataset_D/{i}/color/{pic_name}"
            img = cv2.imread(rgb_dir)
            img2 = cv2.resize(img, (300, 300))
            # b, g, r = cv2.split(img)
            # thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
            img_rgb = img2 / 255.0

            d_dir = f"./Dataset_D/{i}/depth/{pic_name}"
            img_d = cv2.imread(d_dir, cv2.IMREAD_GRAYSCALE)
            img_d = cv2.resize(img_d, (300, 300))
            img_d = img_d / 255.0
            img_d = img_d[:, :, np.newaxis]
            img_rgbd_normalized = np.concatenate([img_rgb, img_d], axis=-1)
            img_rgbd_normalized = np.transpose(img_rgbd_normalized, (2, 0, 1))
            label_value = labels_dict.get(i, 0)
            data.append(img_rgbd_normalized)
            labels.append(label_value)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)


def main():
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    check_path = 'model/CNN/2024-01-13-23-41-AdvanceMobileNetV2/fold_2_epoch_254_test_loss_2.307e-06.pth'
    model = AdvanceMobileNetV2()
    model.load_state_dict(torch.load(check_path), strict=False)
    model.to(device)
    model.eval()

    # Define the input tensors
    X = torch.randn(1, 1, 300, 300).to(device)  # Replace with your input shape
    X.requires_grad = False  # Set to True if you need gradients

    # Pass the input through the model
    with torch.no_grad():
        result = model(X)

    # Initialize real_data (replace with your initialization method)
    real_data =

    # Now you can work with 'result' and 'real_data' in PyTorch

    for duck_num in tqdm(range(1,51),desc='Saving Data',unit='File'):
        dirs = os.listdir(test_pic_dir + str(duck_num))
        dirs = sorted(dirs)
        num = 1
        ws = wb.add_sheet(str(duck_num))
        ws.write(0, 0, '序号')
        ws.write(0, 1, '预测值')
        ws.write(0, 2, '真实值')
        ws.write(0, 3, '误差')
        ws.write(0, 4, 'FPS(推理时间/ms)')

        pre_weight_sum = 0.0  # 用于累积预测值的和
        real_weight_sum = 0.0  # 用于累积真实值的和
        fps_sum = 0.0  # 用于累积真实值的和

        for pic in dirs:
            test_pic = load_data(duck_num, pic)
            test_pic = torch.from_numpy(test_pic).unsqueeze(0).to(device)

            # Measure time for forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            with torch.no_grad():
                a = model(test_pic)
                b = a[0].cpu().numpy()
                pre_weight = b[0]

            end_time.record()
            torch.cuda.synchronize()  # 确保 GPU 完成了前面的所有操作
            elapsed_time = start_time.elapsed_time(end_time)  # milliseconds
            fps = elapsed_time  # Convert milliseconds to seconds

            pre_weight_sum += pre_weight
            real_weight_sum += real_data[duck_num - 1]
            fps_sum += fps

            ws.write(num, 0, int(pic.split('.')[0]))
            ws.write(num, 1, float(pre_weight))
            ws.write(num, 2, real_data[duck_num - 1])
            loss = abs(float(pre_weight) - float(real_data[duck_num - 1]))
            ws.write(num, 3, loss)
            ws.write(num, 4, fps)
            pre_weight = None
            num = num + 1
        # 计算平均值
        average_pre_weight = pre_weight_sum / len(dirs)
        average_real_weight = real_weight_sum / len(dirs)
        ws.write(num, 0, '平均值')
        ws.write(num, 1, average_pre_weight)
        ws.write(num, 2, average_real_weight)
        loss = abs(average_pre_weight - average_real_weight)
        ws.write(num, 3, loss)
        ws.write(num, 4, fps_sum/len(dirs))

        wb.save('单种报告fps01.xls')
    tqdm.write(f'finish ' + str(duck_num) + '/50')


if __name__ == '__main__':
    main()