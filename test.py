import torch


PATH = "./dist_training_model.pth"

from train import MyNet, MyImageDataset, x_transforms, target_transform, classes


my_net = MyNet()
my_net.load_state_dict(torch.load(PATH))

testset = MyImageDataset(
    root="dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test",
    meta_file="dataset/Chest_xray_Corona_Metadata.csv",
    train=False,
    transform=x_transforms,
    target_transform=lambda y: classes.index(y),
)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on test images: %d %%" % (100 * correct / total))
