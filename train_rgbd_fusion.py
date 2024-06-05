import os
import datetime
import torchsummary
import torch.optim as optim
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from nets import get_network
from nets.MobileNetV2 import MobileNetV2
from dataset import MyDataset
from thop import profile
from models.DCF_ResNet_models import DCF_ResNet
from models.fusion import fusion
from models.depth_calibration_models import depth_estimator,discriminator

NUM_FOLDS = 3
BATCH_SIZE = 2
input_size = 352

def train_model(model,model_discriminator,model_estimator,model_rgb,model_d,model_fusion, train_loader, optimizer, criterion, device):
    model.train()
    model_estimator.train()
    model_rgb.train()
    model_d.train()
    model_fusion.train()
    model_discriminator.train()
    accumulated_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(total=num_batches, desc="Training", unit="batch") as t:
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            inputs_rgb = inputs[:, :3, :, :]  # 前三个通道作为inputs_rgb
            inputs_d = inputs[:, 3, :, :].unsqueeze(1)  # 第四个通道作为inputs_d
            atts_rgb, dets_rgb,x3_r,x4_r,x5_r = model_rgb(inputs_rgb)  # model_rgb 的输入是[1, 3, 352, 352]

            score = model_discriminator(inputs_d)
            score = torch.softmax(score,dim=1)
            x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()
            pred_depth = model_estimator(inputs_rgb,x3_, x4_, x5_)
            depth_calibrated = torch.mul(inputs_d,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                         expand(-1, 1, 352, 352)) \
                               + torch.mul(pred_depth,score[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                         expand(-1, 1, 352, 352))

            depths = depth_calibrated
            depths = torch.cat([depths,depths,depths],dim=1)
            atts_depth, dets_depth, x3_d,x4_d,x5_d = model_d(depths)
            # outputs_d = model_d(inputs_d)

            # fusion
            x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
            x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
            att, pred, x3, x4, x5 = model_fusion(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

            res_d = dets_rgb + dets_depth +pred
            input_rgbd = torch.cat((inputs_rgb,res_d), dim=1)

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


def eval_model(model,model_discriminator,model_estimator,model_rgb,model_d,model_fusion, test_data, test_label, criterion, device):
    model.eval()
    model_estimator.eval()
    model_rgb.eval()
    model_d.eval()
    model_fusion.eval()
    model_discriminator.eval()
    num_batches = len(test_label) // BATCH_SIZE + int(len(test_label) % BATCH_SIZE > 0)

    with tqdm(total=num_batches, desc="Evaluating", unit="batch") as t:
        test_outputs_list = []
        with torch.no_grad():
            for j in range(0, len(test_label), BATCH_SIZE):
                test_batch = test_data[j:j + BATCH_SIZE].to(device)

                # inputs, targets = inputs.to(device), targets.to(device)

                inputs_rgb = test_batch[:, :3, :, :]  # 前三个通道作为inputs_rgb
                inputs_d = test_batch[:, 3, :, :].unsqueeze(1)  # 第四个通道作为inputs_d
                atts_rgb, dets_rgb,x3_r,x4_r,x5_r = model_rgb(inputs_rgb)  # model_rgb 的输入是[1, 3, 352, 352]

                score = model_discriminator(inputs_d)
                score = torch.softmax(score,dim=1)
                x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()
                pred_depth = model_estimator(inputs_rgb,x3_, x4_, x5_)
                depth_calibrated = torch.mul(inputs_d,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                            expand(-1, 1, 352, 352)) \
                                + torch.mul(pred_depth,score[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                            expand(-1, 1, 352, 352))

                depths = depth_calibrated
                depths = torch.cat([depths,depths,depths],dim=1)
                atts_depth, dets_depth, x3_d,x4_d,x5_d = model_d(depths)
                # outputs_d = model_d(inputs_d)

                # fusion
                x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
                x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
                att, pred, x3, x4, x5 = model_fusion(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

                res_d = dets_rgb + dets_depth +pred
                input_rgbd = torch.cat((inputs_rgb,res_d), dim=1)

                test_batch_outputs = model(input_rgbd).view(-1)



                # test_outputs_rgb = model_rgb(test_batch)
                # test_outputs_d = model_d(test_batch)
                # test_outputs_fusion = model_fusion(test_outputs_rgb,test_outputs_d)
                # test_outputs = model(test_outputs_fusion)

                # test_batch_outputs = model(outputs).view(-1)
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
    model_estimator = depth_estimator().to(device)
    model_discriminator = discriminator(n_class=2).to(device)
    
    net_name = 'MobileNetV2'
    net = get_network(net_name)
    model = net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
    folder_name = f'/home/chen/Desktop/data/train_data/{timestamp}{"-"}{net_name}/'
    os.makedirs(folder_name, exist_ok=True)
    schedule = StepLR(optimizer, step_size=20, gamma=0.5)

    # Split the dataset into folds for cross-validation
    datasets = MyDataset()
    data_loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
    all_data_img, all_data_labels = zip(*[(data, labels) for data, labels in data_loader])
    data_img = torch.cat(all_data_img, dim=0).permute(0,3,1,2)
    # data_labels = torch.cat(all_data_labels, dim=0).unsqueeze(1)

    # data_img = torch.cat(all_data_img, dim=0).permute(0,3,1,2)
    data_labels = torch.cat(all_data_labels, dim=0).view(-1)

    dataset_folds = torch.chunk(data_img, NUM_FOLDS, dim=0)
    labels_folds = torch.chunk(data_labels, NUM_FOLDS, dim=0)

    # Print model summary using torchsummary
    # print(f'===============>Building {net_name}‘s model<===============')
    # summary_input = (4, input_size, input_size)                                 # 单通道
    # torchsummary.summary(model, input_size=summary_input)

    # Calculate FLOPs using thop
    # input_data = torch.randn(BATCH_SIZE, 4, input_size, input_size).to(device)
    # flops, params = profile(model, inputs=(input_data,))

    # print(f"Number of FLOPs: {flops}")

    for fold in range(NUM_FOLDS):
        test_data = dataset_folds[fold]
        test_label = labels_folds[fold]

        train_data = torch.cat([dataset_folds[i] for i in range(NUM_FOLDS) if i != fold], dim=0)
        train_labels = torch.cat([labels_folds[i] for i in range(NUM_FOLDS) if i != fold], dim=0)

        # 使用MyDataset的数据加载器
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        prev_epoch_loss = float('inf')
        data_list=[]


        for i in range(epochs):
            train_loss = train_model(model, model_discriminator,model_estimator,model_rgb,model_d,model_fusion, train_loader, optimizer, criterion, device)
            test_loss = eval_model(model,model_discriminator,model_estimator,model_rgb,model_d,model_fusion, test_data, test_label, criterion, device)
            schedule.step()

            if test_loss <= prev_epoch_loss:
                prev_epoch_loss = test_loss
                torch.save(model.state_dict(),
                           f'{folder_name}/fold_{fold + 1}_epoch_{i + 1}_test_loss_{round(prev_epoch_loss.item(), 9)}.pth')
            with tqdm(total=epochs, desc=f"Fold {fold + 1}/{NUM_FOLDS}", unit="epoch") as epoch_t:
                epoch_t.set_postfix(Train_Loss=f"{train_loss:.9f}",
                                    Test_Loss=f"{test_loss.item():.9f}",
                                    Prev_Epoch_Loss=f"{prev_epoch_loss.item():.9f}")
                epoch_t.update(i)

                file_name = f'{timestamp}{"-"}{net_name}' + ".json"
                data = {
                    "epoch": f"fold_{fold + 1}_epoch_{i + 1}",
                    "Train_Loss": f"{train_loss:.9f}",
                    "Test_Loss": f"{test_loss.item():.9f}",
                    "Prev_Epoch_Loss": f"{prev_epoch_loss.item():.9f}"
                }
                data_list.append(data)
            sorted_data=sorted(data_list,key=lambda x:float(x["Test_Loss"]))
            sorted_file_name=f'{timestamp}{"-"}{net_name}_sorted.json'
            with open(os.path.join(folder_name,sorted_file_name),'w')as f:
                json.dump(sorted_data,f,indent=4)


if __name__ == '__main__':
    main()
