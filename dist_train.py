import deepspeed
import torch
import torch.utils
import torch.utils.data
import torchvision
import argparse
import os

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torchvision.io import read_image
from PIL import Image


def add_argument() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Distributed training")

    parser.add_argument("--with_cuda", default=False, action="store_true")
    parser.add_argument("--use_ema", default=False, action="store_true")

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser=parser)

    args: argparse.Namespace = parser.parse_args()
    return args


my_transform: transforms.Compose = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# cifar_trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=my_transform)
# trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=32, shuffle=True, num_workers=4)

# cifar_testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=my_transform)
# testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=32, shuffle=False, num_workers=4)

classes = ("Normal", "Pnemonia")


class MyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, meta_file: str, train: bool = True, transform=None, target_transform=None):
        self.image_meta_infos: pd.DataFrame = pd.read_csv(meta_file)
        self.root: str = root
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.image_meta_infos = self.image_meta_infos[self.image_meta_infos["Dataset_type"] == "TRAIN"]
        else:
            self.image_meta_infos = self.image_meta_infos[self.image_meta_infos["Dataset_type"] == "TEST"]

    def __len__(self):
        return len(self.image_meta_infos)

    def __getitem__(self, idx: int):
        img_path: str = os.path.join(self.root, self.image_meta_infos.iloc[idx, 1])
        image: Image.Image = Image.open(img_path).convert("L")
        label: str = self.image_meta_infos.iloc[idx, 2]
        image: torch.Tensor = self.transform(image)
        label: torch.Tensor = self.target_transform(label)
        return image, label


x_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
target_transform = lambda y: torch.zeros(2, dtype=torch.float).scatter_(
    dim=0, index=torch.tensor(classes.index(y)), value=1
)

train_dataset = MyImageDataset(
    root="dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train",
    meta_file="dataset/Chest_xray_Corona_Metadata.csv",
    train=True,
    transform=x_transforms,
    target_transform=target_transform,
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # input 1x256x256
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # Output 16x256x256
        self.pool = nn.MaxPool2d(2, 2)  # Output 16x128x128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # Output 32x128x128
        # Further pooling 32x64x64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Output 64x64x64
        # Further pooling 64x32x32
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # Output 128x32x32
        # Final pooling 128x16x16
        # Flatten 128x16x16 = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 8000)
        self.fc2 = nn.Linear(8000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


my_net = MyNet()
parameters: torch.Tensor = filter(lambda p: p.requires_grad, my_net.parameters())
args = add_argument()
criterion = nn.CrossEntropyLoss()


# init deepspeed
model_engine, optimizer, trainloader, _ = deepspeed.initialize(
    args=args, model=my_net, model_parameters=parameters, training_data=train_dataset
)


for i, data in enumerate(trainloader, 0):
    inputs: torch.Tensor = data[0].to(model_engine.device)
    labels: torch.Tensor = data[1].to(model_engine.device)

    outputs: torch.Tensor = model_engine(inputs)
    loss: torch.Tensor = criterion(outputs, labels)

    model_engine.backward(loss)
    model_engine.step()

print(f'Finished Training')


model_engine.save_checkpoint(save_dir="./dist_training_checkpoint")
torch.save(my_net.state_dict(), "./dist_training_model.pth")


