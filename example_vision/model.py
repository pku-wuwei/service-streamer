# coding=utf-8
# Created by Meteorix at 2019/8/9

import torch
from torchvision import models, transforms, datasets

device = "cuda"
model = models.densenet121(pretrained=True).to(device)
model.eval()


def get_data_loader(batch_size=4, num_workers=2):
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # dataset = datasets.ImageFolder(root='./pic', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader


def batch_predict(batched_data):
    prediction = model(batched_data.to(device))
    _, y_hat = prediction.max(1)
    predicted_ids = y_hat.tolist()
    print(predicted_ids)


if __name__ == "__main__":
    loader = get_data_loader()
    for i, (data, labels) in enumerate(loader):
        batch_predict(data)
