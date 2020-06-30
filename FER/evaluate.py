import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train import Network, test

VAL_DATA_FOLDER = './output/PrivateTest'
print(VAL_DATA_FOLDER)
# model_path = './checkpoints/resnet18/2020-05-28-18-41/model_best_by_accu.pth'
model_path = './checkpoints/resnet18/2020-06-09-21-49/model_best_by_accu.pth'
print(model_path)

BATCH_SIZE = 128
N_CPU = 4
classes_name = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    ])
    val_data = datasets.ImageFolder(VAL_DATA_FOLDER, transform=transform)
    val_dataloader = DataLoader(val_data, \
                                batch_size=BATCH_SIZE,\
                                shuffle=True, \
                                pin_memory=True)           
    # if model_path.split('/')[-2] == '
    model = Network()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # model.to(device)
    test(model, val_dataloader)
    print('done!')
    # sudo scp -P 2205 -r src cv@10.100.53.68:/home/cv/phat/CenterNet