import cv2
import csv
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_dir='/home/chen/Desktop/data/Dataset_D', label_file='data_label.csv'):
        self.data_dir = data_dir
        self.label_file = label_file
        self.labels_dict = self.load_labels()
        self.image_paths = self.load_image_paths()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((352, 352)),
            transforms.ToTensor()
        ])

    def load_labels(self):
        labels_dict = {}
        try:
            with open(self.label_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # 跳过标题行
                labels_dict = {int(row[0]): float(row[1]) for row in csv_reader}
        except FileNotFoundError:
            print(f"错误: 文件 '{self.label_file}' 未找到。")
        return labels_dict

    def load_image_paths(self):
        image_paths = []
        for i in tqdm(range(1, 51), desc='加载图片路径', unit='图片集'):
            rgb_dir = os.path.join(self.data_dir, f"{i}/color")
            depth_dir = os.path.join(self.data_dir, f"{i}/depth")

            if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
                print(f"错误: 目录 '{rgb_dir}' 或 '{depth_dir}' 未找到。")
                continue

            rgb_files = sorted(os.listdir(rgb_dir))
            depth_files = sorted(os.listdir(depth_dir))

            for rgb_file, depth_file in zip(rgb_files, depth_files):
                rgb_path = os.path.join(rgb_dir, rgb_file)
                depth_path = os.path.join(depth_dir, depth_file)
                image_paths.append((rgb_path, depth_path, i))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb_path, depth_path, label_index = self.image_paths[idx]
        
        img_rgb = cv2.imread(rgb_path).astype(np.float32)  # 转换为单精度浮点型
        img_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # 转换为单精度浮点型
        
        if img_rgb is None or img_depth is None:
            raise ValueError(f"错误: 无法读取 '{rgb_path}' 或 '{depth_path}'。")

        img_rgb = cv2.resize(img_rgb, (352, 352))
        img_depth = cv2.resize(img_depth, (352, 352))

        # 创建4通道图像
        concatenated_image = np.zeros((352, 352, 4), dtype=np.uint8)
        concatenated_image[:, :, :3] = img_rgb  # RGB通道
        concatenated_image[:, :, 3] = img_depth  # 深度通道

        img_transformed = self.transform(concatenated_image)

        label_value = self.labels_dict.get(label_index, -1)  # 使用-1表示缺少标签
        if label_value == -1:
            raise ValueError(f"错误: 未找到索引 {label_index} 的标签。")

        return img_transformed, np.float32(label_value)
