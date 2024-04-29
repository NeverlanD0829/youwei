import os
import datetime
import torchsummary
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from nets import get_network
from nets.MobileNetV2 import *
from dataset import MyDataset
from thop import profile

NUM_FOLDS = 3
BATCH_SIZE = 64
input_size = 300

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    accumulated_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(total=num_batches, desc="Training", unit="batch") as t:
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.item()
            torch.cuda.empty_cache()
            t.update(1)
    train_loss = accumulated_loss / num_batches
    return train_loss


def eval_model(model, test_data, test_label, criterion, device):
    model.eval()
    num_batches = len(test_label) // BATCH_SIZE + int(len(test_label) % BATCH_SIZE > 0)

    with tqdm(total=num_batches, desc="Evaluating", unit="batch") as t:
        test_outputs_list = []
        with torch.no_grad():
            for j in range(0, len(test_label), BATCH_SIZE):
                test_batch = test_data[j:j + BATCH_SIZE].to(device)
                test_batch_outputs = model(test_batch)
                test_outputs_list.append(test_batch_outputs)
                t.update(1)

        test_outputs = torch.cat(test_outputs_list)

    test_loss = criterion(test_outputs, test_label.to(device))
    return test_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_name = 'MobileNetV2'
    net = get_network(net_name)
    model = net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 300
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
    folder_name = f'model/CNN/{timestamp}{"-"}{net_name}/'
    os.makedirs(folder_name, exist_ok=True)
    schedule = StepLR(optimizer, step_size=40, gamma=0.5)

    # Split the dataset into folds for cross-validation
    datasets = MyDataset()
    data_loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
    all_data_img, all_data_labels = zip(*[(data, labels) for data, labels in data_loader])
    data_img = torch.cat(all_data_img, dim=0).unsqueeze(1)
    data_labels = torch.cat(all_data_labels, dim=0).unsqueeze(1)

    dataset_folds = torch.chunk(data_img, NUM_FOLDS, dim=0)
    labels_folds = torch.chunk(data_labels, NUM_FOLDS, dim=0)

    # Print model summary using torchsummary
    print(f'===============>Building {net_name}‘s model<===============')
    summary_input = (1, input_size, input_size)                                 # 单通道
    torchsummary.summary(model, input_size=summary_input)

    # Calculate FLOPs using thop
    input_data = torch.randn(BATCH_SIZE, 1, input_size, input_size).to(device)
    flops, params = profile(model, inputs=(input_data,))

    print(f"Number of FLOPs: {flops}")

    for fold in range(NUM_FOLDS):
        test_data = dataset_folds[fold]
        test_label = labels_folds[fold]

        train_data = torch.cat([dataset_folds[i] for i in range(NUM_FOLDS) if i != fold], dim=0)
        train_labels = torch.cat([labels_folds[i] for i in range(NUM_FOLDS) if i != fold], dim=0)

        # 使用MyDataset的数据加载器
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        prev_epoch_loss = float('inf')

        for i in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            test_loss = eval_model(model, test_data, test_label, criterion, device)
            schedule.step()

            if test_loss <= prev_epoch_loss:
                prev_epoch_loss = test_loss
                torch.save(model.state_dict(),
                           f'{folder_name}/fold_{fold + 1}_epoch_{i + 1}_test_loss_{round(prev_epoch_loss.item(), 9)}.pth')
            with tqdm(total=epochs, desc=f"Fold {fold + 1}/{NUM_FOLDS}", unit="epoch") as epoch_t:
                epoch_t.set_postfix(Train_Loss=f"{train_loss:.6f}",
                                    Test_Loss=f"{test_loss.item():.6f}",
                                    Prev_Epoch_Loss=f"{prev_epoch_loss.item():.6f}")
                epoch_t.update(i)


if __name__ == '__main__':
    main()
