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
from models.DCF_ResNet_models import DCF_ResNet
from models.fusion import fusion


test_pic_dir = "/home/chen/Desktop/data/Dataset_D/"

def main():
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load the model checkpoint
    model = MobileNetV2().to(device)
    model_rgb = DCF_ResNet().to(device)
    model_d = DCF_ResNet().to(device)
    model_fusion = fusion().to(device)

    model.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-11-18-46-MobileNetV2/Number 123 epoch/model_weight_9.254e-06.pth"), strict=False)
    model_rgb.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-11-18-46-MobileNetV2/Number 123 epoch/model_rgb_weight_9.254e-06.pth"), strict=False)
    model_d.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-11-18-46-MobileNetV2/Number 123 epoch/model_d_weight_9.254e-06.pth"), strict=False)
    model_fusion.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-11-18-46-MobileNetV2/Number 123 epoch/model_fusion_weight_9.254e-06.pth"), strict=False)
    
    model.eval()
    model_rgb.eval()
    model_d.eval()
    model_fusion.eval()

    # Initialize real_data (replace with your initialization method)
    real_weight = []
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((352, 352)),
        transforms.ToTensor()
    ])

    labels_csv_file = 'data_label.csv'
    try:
        with open(labels_csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            labels_dict = {int(row[0]): float(row[1]) for row in csv_reader}
    except FileNotFoundError:
        print(f"Error: File '{labels_csv_file}' not found.")
        return

    for sample_num in tqdm(range(1, 56), desc='Saving Data', unit='File'):
        num = 1
        ws = wb.add_sheet(str(sample_num))
        ws.write(0, 0, '序号')
        ws.write(0, 1, '预测值')
        ws.write(0, 2, '真实值')
        ws.write(0, 3, '误差')
        ws.write(0, 4, 'FPS(推理时间/ms)')

        pre_weight_sum = 0.0  # 用于累积预测值的和
        fps_sum = 0.0  # 用于累积FPS的和

        label_value = labels_dict.get(sample_num)
        real_weight.append(label_value)

        rgb_dir = f"/home/chen/Desktop/data/Dataset_D/{sample_num}/color"
        depth_dir = f"/home/chen/Desktop/data/Dataset_D/{sample_num}/depth"
        rgb_files = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))
        num_images = len(rgb_files)

        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files), start=1):
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, depth_file)

            img_rgb = cv2.imread(rgb_path)
            img_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            if img_rgb is None or img_depth is None:
                raise ValueError(f"错误: 无法读取 '{rgb_dir}' 或 '{depth_dir}'。")

            img_rgb = cv2.resize(img_rgb, (352, 352))
            img_depth = cv2.resize(img_depth, (352, 352))
            # 创建4通道图像
            concatenated_image = np.zeros((352, 352, 4), dtype=np.uint8)
            concatenated_image[:, :, :3] = img_rgb  # RGB通道
            concatenated_image[:, :, 3] = img_depth  # 深度通道

            img_transformed = transform(concatenated_image)
            img_transformed = img_transformed.unsqueeze(0)

            # Measure time for forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            with torch.no_grad():
                inputs_rgb = img_transformed[:, :3, :, :].to(device)  # 前三个通道作为inputs_rgb
                inputs_d = img_transformed[:, 3, :, :].unsqueeze(1).to(device)  # 第四个通道作为inputs_d
                atts_rgb, dets_rgb, x3_r, x4_r, x5_r = model_rgb(inputs_rgb)  # model_rgb 的输入是[1, 3, 352, 352]

                depths = torch.cat([inputs_d, inputs_d, inputs_d], dim=1)
                atts_depth, dets_depth, x3_d, x4_d, x5_d = model_d(depths)

                # fusion
                x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
                x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
                att, pred, x3, x4, x5 = model_fusion(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

                res_d = dets_rgb + dets_depth + pred
                input_rgbd = torch.cat((inputs_rgb, res_d), dim=1)

                a = model(input_rgbd)
                pre_weight = a[0].item()

            end_time.record()
            torch.cuda.synchronize()  # 确保 GPU 完成了前面的所有操作
            elapsed_time = start_time.elapsed_time(end_time)  # milliseconds

            pre_weight_sum += pre_weight
            fps_sum += elapsed_time

            ws.write(num, 0, i)
            ws.write(num, 1, pre_weight)
            ws.write(num, 2, label_value)
            loss = abs(pre_weight - label_value)
            ws.write(num, 3, loss)
            ws.write(num, 4, elapsed_time)
            num += 1

        # 计算平均值
        average_pre_weight = pre_weight_sum / num_images
        average_fps = fps_sum / num_images

        ws.write(num, 0, '平均值')
        ws.write(num, 1, average_pre_weight)
        ws.write(num, 2, label_value)
        loss = abs(average_pre_weight - label_value)
        ws.write(num, 3, loss)
        ws.write(num, 4, average_fps)

        wb.save('报告RGBD_dcf_test.xls')
        tqdm.write(f'finish ' + str(sample_num) + '/55')


if __name__ == '__main__':
    main()
