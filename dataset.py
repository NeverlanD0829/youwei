import cv2
import csv
import numpy as np
import os
from tqdm import tqdm
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
        
        # rgb图片训练
        # for i in tqdm(range(1, 51), desc='Load Image', unit='Image'):
        #     dirs = os.listdir(f"/home/chen/Desktop/data/Dataset/{i}")
        #     for pic_name in dirs:
        #         pic_dir = f"/home/chen/Desktop/data/Dataset/{i}/{pic_name}"
        #         img = cv2.imread(pic_dir)
        #         img = cv2.resize(img, (300, 300))
        #         b, g, r = cv2.split(img)
        #         thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
        #         img_normalized = img2 / 255.0
        #         label_index = i
        #         label_value = labels_dict.get(label_index)
        #         data.append(img_normalized)
        #         labels.append(label_value)
        # return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)
    
     # rgb_d图片训练
        data = []
        labels = []

        for i in tqdm(range(1, 51), desc='Load Image', unit='Image'):
            rgb_dir = f"/home/chen/Desktop/data/Dataset_D/{i}/color"
            depth_dir = f"/home/chen/Desktop/data/Dataset_D/{i}/depth"

            rgb_files = os.listdir(rgb_dir)
            depth_files = os.listdir(depth_dir)

            for j in range(1, 176):
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

                label_index = i
                label_value = labels_dict.get(label_index)

                data.append(img_normalized)
                labels.append(label_value)

        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


