import numpy as np
import cv2
import xlwt
import xlrd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.models as models
from nets.Wpdnet import *
from nets.Resnet18 import *
from nets.Resnet34 import *

test_pic_dir = "./Dataset/"

def pic_init(num, pic):                                 #读取测试图片，将其调整大小并提取红色通道作为输入
    image_dir = test_pic_dir + str(num) + '/' + pic
    # print(image_dir)
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (250, 250))
    b, g, r = cv2.split(image)
    predict_data = []
    predict_data.append(r)
    predict_data = np.array(predict_data)
    predict_data = predict_data.astype(np.float32)
    return predict_data

def real_data_init():                                   #它从Excel文件中读取真实的鸭子体重数据
    workbook = xlrd.open_workbook('Weight_Data.xls')
    Data_sheet = workbook.sheets()[0]
    rowNum = Data_sheet.nrows
    colNum = Data_sheet.ncols
    xls_list = []
    for i in range(rowNum):
        rowlist = []
        for j in range(colNum):
            rowlist.append(Data_sheet.cell_value(i, j))
        xls_list.append(rowlist)
    real_data = []
    for i in range(1,51):
        c = int((i-1)/10)
        r = (i-1)%10 + 2
        real_data.append(float(xls_list[r][c]))
    return real_data

def main():                                            #恢复了训练好的模型，并通过TensorFlow的会话进行预测。它遍历了每种鸭子的图片文件夹，并对每张图片进行预测。预测结果被写入Excel文件中，并计算了预测值与真实值之间的误差
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')
    wb = xlwt.Workbook()


    # Create a PyTorch session (not exactly the same as TensorFlow session)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    check_path = 'model/CNN/2023-10-09-20-50-Resnet34/epoch=856,test_loss=9.5e-05.pth'
    model = Resnet34()                                               # Replace with your model definition
    model.load_state_dict(torch.load(check_path))
    model.to(device)
    model.eval()

    # Define the input tensors
    X = torch.randn(1, 1, 250, 250).to(device)  # Replace with your input shape
    X.requires_grad = False  # Set to True if you need gradients


    # Pass the input through the model
    with torch.no_grad():
        result = model(X)

    # Initialize real_data (replace with your initialization method)
    real_data = real_data_init()

    # Now you can work with 'result' and 'real_data' in PyTorch

    for duck_num in range(1,51):
        dirs = os.listdir(test_pic_dir + str(duck_num))
        dirs = sorted(dirs)
        num = 1
        ws = wb.add_sheet(str(duck_num))
        ws.write(0, 0, '序号')
        ws.write(0, 1, '预测值')
        ws.write(0, 2, '真实值')
        ws.write(0, 3, '误差')
        for pic in dirs:
            test_pic = pic_init(duck_num, pic)
            test_pic = torch.from_numpy(test_pic).unsqueeze(0).to(device)
            with torch.no_grad():
                a = model(test_pic)
                b = a[0].cpu().numpy()
                pre_weight = b[0]
            ws.write(num, 0, int(pic.split('.')[0]))
            ws.write(num, 1, float(pre_weight))
            ws.write(num, 2, real_data[duck_num - 1])
            loss = abs(float(pre_weight) - float(real_data[duck_num - 1]))
            ws.write(num, 3, loss)
            pre_weight = None
            num = num + 1
        print('finish ' + str(duck_num) + '/50!')

        wb.save('原始模型单种报告1.xls')
        print('Save the report.xls')

if __name__ == '__main__':
    main()