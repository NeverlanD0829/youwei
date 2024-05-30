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

        for i in tqdm(range(1, 51), desc='Load Image', unit='Image'):
            dirs = os.listdir(f"/home/chen/Desktop/data/Dataset/{i}")
            for pic_name in dirs:
                pic_dir = f"/home/chen/Desktop/data/Dataset/{i}/{pic_name}"
                img = cv2.imread(pic_dir)
                img = cv2.resize(img, (300, 300))
                b, g, r = cv2.split(img)
                thresh, img2 = cv2.threshold(g, 90, 0, cv2.THRESH_TOZERO)
                img_normalized = img2 / 255.0
                label_index = i
                label_value = labels_dict.get(label_index)
                data.append(img_normalized)
                labels.append(label_value)
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


