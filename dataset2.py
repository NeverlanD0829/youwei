import cv2
import csv
import numpy as np
import xlrd
import random
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self):
        self.data, self.labels = self.load_data()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor()])

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
                img_rgbd_normalized=np.transpose(img_rgbd_normalized, (2, 0, 1))
                # print(img_rgbd_normalized.shape)
                label_value = labels_dict.get(i, 0)
                data.append(img_rgbd_normalized)
                labels.append(label_value)
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#
# if __name__ == "__main__":
#     dataset = MyDataset()



