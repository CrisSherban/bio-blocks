import os
import copy
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch import optim
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

covid_dataset_path = '../datasets/chest_xray'
classes = ['no-covid', 'covid']


class CovDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, phase=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.df.drop(self.df[self.df.view != 'PA'].index, inplace=True)
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.df['finding'].iloc[idx] != 'COVID-19':
            finding = 0
            img_path = os.path.sep.join([covid_dataset_path, 'images', self.df['filename'].iloc[idx]])
            image = Image.open(img_path)
            sample = {'image': image, 'finding': finding}

            if self.transform:
                sample = {'image': self.transform[self.phase](sample['image']), 'finding': finding}

        else:
            finding = 1
            img_path = os.path.sep.join([covid_dataset_path, 'images', self.df['filename'].iloc[idx]])
            image = Image.open(img_path)
            sample = {'image': image, 'finding': finding}

            if self.transform:
                sample = {'image': self.transform[self.phase](sample['image']), 'finding': finding}

        return sample


class HistEqualization(object):
    def __call__(self, image):
        return ImageOps.equalize(image, mask=None)


class SmoothImage(object):
    def __call__(self, image):
        return image.filter(ImageFilter.SMOOTH_MORE)


def train(model, criterion, optimizer, dataset_sizes, dataloaders, num_epochs, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs = torch.FloatTensor(data['image'])
                labels = torch.LongTensor(data['finding'])
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best validation Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomRotation(30, fill=(0,)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            HistEqualization(),
            SmoothImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            HistEqualization(),
            SmoothImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25])
        ]),
    }

    image_datasets = {
        x: CovDataset(
            csv_path=os.path.sep.join([covid_dataset_path, 'metadata.csv']),
            root_dir=covid_dataset_path,
            transform=data_transforms,
            phase=x)
        for x in ['train', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=32,
                                       shuffle=True if x == 'train' else False,
                                       num_workers=4)
        for x in ['train', 'test']
    }

    dataset_sizes = {
        x: len(image_datasets[x])
        for x in ['train', 'test']
    }

    device = torch.device("cpu")

    model = models.resnet18(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(fc_features, fc_features // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(fc_features // 2, 2)
    )
    model.conv1.requires_grad_()
    model.fc.requires_grad_()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transfer_learned_model = train(model,
                                   criterion,
                                   optimizer,
                                   dataset_sizes,
                                   dataloaders,
                                   num_epochs=5,
                                   device=device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    fine_tuned_model = train(transfer_learned_model,
                             criterion,
                             optimizer,
                             dataset_sizes,
                             dataloaders,
                             num_epochs=5,
                             device=device)

    model_scripted = torch.jit.script(model)
    model_scripted.save('../models/model_torch_script.pt')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            inputs = data['image']
            labels = data['finding']
            outputs = fine_tuned_model(inputs.float().to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print(f'Accuracy of the network: {100 * correct / total}')

    dataloader = torch.utils.data.DataLoader(
        image_datasets['test'],
        batch_size=(len(image_datasets['test'])),
        num_workers=4)

    dataiter = iter(dataloader)
    data = next(dataiter)
    images = torch.FloatTensor(data['image'])
    labels = torch.LongTensor(data['finding'])

    fine_tuned_model.to('cpu')
    output = torch.tensor(fine_tuned_model(images).detach().numpy())

    from sklearn.metrics import classification_report
    report = classification_report(labels,
                                   np.argmax(output, 1),
                                   target_names=classes)
    print(report)
    with open("../out/report.txt", "w+") as f:
        f.writelines(report)


if __name__ == "__main__":
    main()
