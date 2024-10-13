import os
import datetime
import torch
import torch.optim as optim
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from nets import get_network
from nets.MobileNetV2 import MobileNetV2
from dataset import MyDataset
from thop import profile
from models.DCF_ResNet_models import DCF_ResNet
# from models.DCF_MobileNetv2_models import DCF_MobileNetV2
from models.fusion import fusion
from models.depth_calibration_models import depth_estimator, discriminator

BATCH_SIZE = 3
input_size = 352

def train_model(model, model_rgb, model_d, model_fusion, train_loader, optimizer, criterion, device):
    model.train()
    model_rgb.train()
    model_d.train()
    model_fusion.train()
    accumulated_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(total=num_batches, desc="Training", unit="batch") as t:
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            inputs_rgb = inputs[:, :3, :, :]                                            # 前三个通道作为inputs_rgb
            inputs_d = inputs[:, 3, :, :].unsqueeze(1)                                  # 第四个通道作为inputs_d
            atts_rgb, dets_rgb, x3_r, x4_r, x5_r = model_rgb(inputs_rgb)                # model_rgb 的输入是[1, 3, 352, 352]

            depths = torch.cat([inputs_d, inputs_d, inputs_d], dim=1)
            atts_depth, dets_depth, x3_d, x4_d, x5_d = model_d(depths)

            # fusion
            x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
            x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
            att, pred, x3, x4, x5 = model_fusion(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

            res_d = dets_rgb + dets_depth + pred
            input_rgbd = torch.cat((inputs_rgb, res_d), dim=1)

            outputs = model(input_rgbd)
            outputs = outputs.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.item()
            torch.cuda.empty_cache()
            t.update(1)
    train_loss = accumulated_loss / num_batches
    return train_loss

def eval_model(model, model_rgb, model_d, model_fusion, test_data, test_label, criterion, device):
    model.eval()
    # model_estimator.eval()
    model_rgb.eval()
    model_d.eval()
    model_fusion.eval()
    # model_discriminator.eval()
    num_batches = len(test_label) // BATCH_SIZE + int(len(test_label) % BATCH_SIZE > 0)

    with tqdm(total=num_batches, desc="Evaluating", unit="batch") as t:
        test_outputs_list = []
        with torch.no_grad():
            for j in range(0, len(test_label), BATCH_SIZE):
                test_batch = test_data[j:j + BATCH_SIZE].to(device)

                inputs_rgb = test_batch[:, :3, :, :]  # 前三个通道作为inputs_rgb
                inputs_d = test_batch[:, 3, :, :].unsqueeze(1)  # 第四个通道作为inputs_d
                atts_rgb, dets_rgb, x3_r, x4_r, x5_r = model_rgb(inputs_rgb)

                depths = torch.cat([inputs_d, inputs_d, inputs_d], dim=1)
                atts_depth, dets_depth, x3_d, x4_d, x5_d = model_d(depths)

                # fusion
                x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
                x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
                att, pred, x3, x4, x5 = model_fusion(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

                res_d = dets_rgb + dets_depth + pred
                input_rgbd = torch.cat((inputs_rgb, res_d), dim=1)

                test_batch_outputs = model(input_rgbd).view(-1)
                test_outputs_list.append(test_batch_outputs)
                t.update(1)

        test_outputs = torch.cat(test_outputs_list)

    test_loss = criterion(test_outputs, test_label.to(device))
    return test_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_rgb = DCF_ResNet().to(device)
    model_d = DCF_ResNet().to(device)
    model_fusion = fusion().to(device)
    # model_estimator = depth_estimator().to(device)
    # model_discriminator = discriminator(n_class=2).to(device)
    
    # 预训练权重
    model_rgb.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-14/Number_40_epoch/model_rgb_weight_3.8047e-05.pth"))
    model_d.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-14/Number_40_epoch/model_d_weight_3.8047e-05.pth"))
    model_fusion.load_state_dict(torch.load("/home/chen/Desktop/data/train_data/2024-06-14/Number_40_epoch/model_fusion_weight_3.8047e-05.pth"))


    net_name = 'MobileNetV2'
    net = get_network(net_name)
    model = net().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(list(model.parameters()) + 
                       list(model_rgb.parameters()) + 
                       list(model_d.parameters()) + 
                       list(model_fusion.parameters()), lr=0.001)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters in main model: {count_parameters(model)}')
    print(f'Total parameters in model_rgb: {count_parameters(model_rgb)}')
    print(f'Total parameters in model_d: {count_parameters(model_d)}')
    print(f'Total parameters in model_fusion: {count_parameters(model_fusion)}')
    print(f'Total parameters in optimizer: {sum(p.numel() for p in optimizer.param_groups[0]["params"])}')

    epochs = 400
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
    folder_name = f'/home/chen/Desktop/data/train_data/{timestamp}{"-"}{net_name}'
    # os.makedirs(folder_name, exist_ok=True)
    schedule = StepLR(optimizer, step_size=40, gamma=0.5)

    datasets = MyDataset()
    data_loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
    all_data_img, all_data_labels = zip(*[(data, labels) for data, labels in data_loader])
    data_img = torch.cat(all_data_img, dim=0)
    data_labels = torch.cat(all_data_labels, dim=0).view(-1)

    # Split data into training and testing sets (e.g., 80% training, 20% testing)
    split_idx = int(0.8 * len(data_img))
    train_data, test_data = data_img[:split_idx], data_img[split_idx:]
    train_labels, test_label = data_labels[:split_idx], data_labels[split_idx:]

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    prev_epoch_loss = float('inf')
    data_list = []

    for i in range(epochs):
        train_loss = train_model(model,  model_rgb, model_d, model_fusion, train_loader, optimizer, criterion, device)
        test_loss = eval_model(model, model_rgb, model_d, model_fusion, test_data, test_label, criterion, device)
        schedule.step()

        if test_loss <= prev_epoch_loss:
            prev_epoch_loss = test_loss
            directory_name = f"Number {i} epoch"
            save_folder = f'{folder_name}/{directory_name}'
            os.makedirs(save_folder, exist_ok=True)
            torch.save(model.state_dict(),f'{folder_name}/{directory_name}/model_weight_{round(prev_epoch_loss.item(), 9)}.pth')
            torch.save(model_rgb.state_dict(),f'{folder_name}/{directory_name}/model_rgb_weight_{round(prev_epoch_loss.item(), 9)}.pth')
            torch.save(model_d.state_dict(),f'{folder_name}/{directory_name}/model_d_weight_{round(prev_epoch_loss.item(), 9)}.pth')
            torch.save(model_fusion.state_dict(),f'{folder_name}/{directory_name}/model_fusion_weight_{round(prev_epoch_loss.item(), 9)}.pth')

        with tqdm(total=epochs, desc="Epoch", unit="epoch") as epoch_t:
            epoch_t.set_postfix(Train_Loss=f"{train_loss:.9f}",
                                Test_Loss=f"{test_loss.item():.9f}",
                                Prev_Epoch_Loss=f"{prev_epoch_loss.item():.9f}")
            epoch_t.update(i)

            file_name = f'{timestamp}{"-"}{net_name}.json'
            data = {
                "epoch": f"epoch_{i + 1}",
                "Train_Loss": f"{train_loss:.9f}",
                "Test_Loss": f"{test_loss.item():.9f}",
                "Prev_Epoch_Loss": f"{prev_epoch_loss.item():.9f}"
            }
            data_list.append(data)

        sorted_data = sorted(data_list, key=lambda x: float(x["Test_Loss"]))
        sorted_file_name = f'{timestamp}{"-"}{net_name}_sorted.json'
        with open(os.path.join(folder_name, sorted_file_name), 'w') as f:
            json.dump(sorted_data, f, indent=4)

if __name__ == '__main__':
    main()
