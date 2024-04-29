import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from nets.MobileNetV2 import *



def predict(image_dir):
    image = cv2.imread(image_dir)
    # image = cv2.resize(image, (300, 300))
    # g = image[:, :, 1]  # 提取通道数据
    # thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
    # predict_data = np.array([img2/255], dtype=np.float32)  # 构建预测数据
    # return predict_data
    image = cv2.resize(image, (300, 300))
    b, g, r = cv2.split(image)
    thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
    predict_data = []
    predict_data.append(img2 / 255)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def main():
    # 创建模型和加载权重
    model = MobileNetV2()
    model.load_state_dict(torch.load('model/CNN/2023-12-26-22-20-MobileNetV2/fold_3_epoch_30_test_loss_6.66e-07.pth'), strict=False)
    model.eval()

    while True:
        image_dir = input("请输入您需要预测的番茄的代号(输入数字‘0’结束程序):")
        if image_dir == '0':
            break

        try:
            predict_data = predict(image_dir)
            predict_tensor = torch.tensor(predict_data)
            predict_tensor=torch.unsqueeze(predict_tensor,1)
            # 使用模型进行预测
            with torch.no_grad():
                prediction = model(predict_tensor)

                print("该番茄预计重：" + str(prediction.item()) + 'kg')
        except:
            print('Open Error! Try again!')
            continue

if __name__ == '__main__':
    main()
