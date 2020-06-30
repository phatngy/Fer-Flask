import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
import numpy as np
from misc import AverageMeter

TRAIN_DATA_FOLDER = './output/Training'
VAL_DATA_FOLDER = './output/PublicTest'
MODEL_PATH = './checkpoints/resnet18/2020-05-14-11-33'
DEFAULT_LEARN_RATE = 1e-4
N_CLASSES = 6
BATCH_SIZE = 128
N_CPU = 4
N_EPOCHS = 100
model_name = 'resnet18'
classes_name = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self._model = models.resnet18(pretrained=True)
        self._model.fc = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        return self._model(x)


def train(model, train_dataloader, val_dataloader, n_epoch, try_to_reload=True):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARN_RATE)

    if try_to_reload:
        try:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.pth')))
        except Exception as e:
            print('Can not load state for model')
            print(e)
        # try:
        #     optimizer.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'optimizer.pth')))
        # except Exception as e:
        #     print('Can not load state for optimizer')
        #     print(e)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=5)

    ### tensorboard
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join('./checkpoints', model_name, time_str)
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float('inf')
    best_accu = 0
    for epoch in range(n_epoch):
        preds_list = []
        labels_list = []

        model.train()
        n_batch = len(train_dataloader)
        pbar = tqdm(desc='Epoch ' + str(epoch), total=n_batch)
        total_loss = AverageMeter()
        total_accu = AverageMeter()
        start_time = time.time()
        for step, data in enumerate(train_dataloader):
            images, label = data
            images = images.to(device)
            label = label.to(device)

            logit = model(images)

            # LOSS FUNCTIONS
            # cross-entropy loss work on raw logits
            loss = F.cross_entropy(logit, label)

            # OPTIMIZATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # INFO
            pbar.update(1)
            loss_value = loss.item()
            total_loss.update(loss_value, images.size(0))
            pbar.set_postfix(Loss=total_loss.avg)

            prediction = torch.argmax(logit, dim=1)
            accu = torch.eq(prediction, label).sum().item() / label.shape[0]
            total_accu.update(accu, images.size(0))
            
            #
            # labels = label.tolist()
            # # print('labels' , labels)
            # pred_batch = prediction.tolist()
            # print('preds', pred_batch)
            preds_list.extend(prediction.tolist())
            labels_list.extend(label.tolist())
            #

            # if step % 250 == 0:
            #     # SAVE TEMPORARY MODELS
            #     torch.save(model.state_dict(), 'model.pth')
            #     torch.save(optimizer.state_dict(), 'optimizer.pth')
            #     # SAVE TENSORBOARD INFO
            #     writer.add_scalar('Minibatch Loss/Train_loss',
            #                       total_loss / (step + 1), epoch * n_batch + step)
            #     writer.add_scalar('Minibatch Accuracy/Accuracy',
            #                       total_accu / (step + 1), epoch * n_batch + step)
        pbar.close()
        end_time = time.time()
        total_time = end_time - start_time

        # labels_list = torch.FloatTensor(labels_list)
        # preds_list = torch.FloatTensor(preds_list)
        train_f1 = f1_score(labels_list, preds_list, average='macro')
        train_recall = recall_score(labels_list, preds_list, average='macro')
        train_confussion = confusion_matrix(labels_list, preds_list)
        # VALIDATION
        val_loss = AverageMeter()
        val_accu = AverageMeter()
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            n_val_batch = len(val_dataloader)
            for step, data in enumerate(val_dataloader):
                images, label = data
                images = images.to(device)
                label = label.to(device)

                logit = model(images)

                # LOSS FUNCTIONS
                # cross-entropy loss work on raw logits
                loss = F.cross_entropy(logit, label)

                # INFO
                pbar.update(1)
                loss_value = loss.item()
                val_loss.update(loss_value, images.size(0))

                prediction = torch.argmax(logit, dim=1)
                accu = torch.eq(prediction, label).sum().item() / label.shape[0]
                val_accu.update(accu, images.size(0))

                # labels_val = label.tolist()
                val_labels.extend(label.tolist())
                val_preds.extend(prediction.tolist())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')
        val_report = classification_report(val_labels, val_preds,target_names=classes_name)
        print('- Train loss:', total_loss.avg,
              '\n - Train accuracy:', total_accu.avg,
              '\n - Train time:', total_time,
              '\n - Train f1_score: ', train_f1,
              '\n - Train recall: ', train_recall,
              '\n - Val loss:', val_loss.avg,
              '\n - Val accuracy:', val_accu.avg,
              '\n - Val F1; ', val_f1,
              '\n - Val recall: ', val_recall,
              '\n - Report: \n', val_report,
              )

        # LEARN RATE ADJUSTMENT
        scheduler.step(val_loss.avg)
        lr = optimizer.param_groups[0]['lr']
        # SAVE TEMPORARY MODELS
        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer.pth'))

        # SAVE BEST MODEL
        current_loss = val_loss.avg
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_best_by_loss.pth'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer_best_by_loss.pth'))
            print('Save best model (loss)')

        current_accu = val_accu.avg
        if current_accu > best_accu:
            best_accu = current_accu
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_best_by_accu.pth'))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer_best_by_accu.pth'))
            print('Save best model (accu)')

        # SAVE TENSORBOARD INFO
        writer.add_scalar('learning rate', lr, epoch)
        writer.add_scalar('Loss/Train_loss', total_loss.avg, epoch)
        writer.add_scalar('Loss/Val_loss', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/Train accuracy', total_accu.avg, epoch)
        writer.add_scalar('Accuracy/Val accuracy', val_accu.avg, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        # DEBUG INFORMATION
        pass

    writer.close()


def test(model, test_dataloader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    n_batch = len(test_dataloader)
    pbar = tqdm(desc='Testing', total=n_batch)
    
    total_loss = AverageMeter()
    total_accu = AverageMeter()
    tic = time.time()
    
    test_labels = []
    test_preds = []
    
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            images, label = data
            images = images.to(device)
            label = label.to(device)

            logit = model(images)

            # LOSS FUNCTIONS
            # cross-entropy loss work on raw logits
            loss = F.cross_entropy(logit, label)

            # INFO
            pbar.update(1)
            loss_value = loss.item()
            # total_loss += loss_value

            prediction = torch.argmax(logit, dim=1)
            acc = torch.eq(prediction, label).sum().item() / images.size(0)
            total_accu.update(acc, images.size(0))
            total_loss.update(loss_value, images.size(0))
            
            test_labels.extend(label.tolist())
            test_preds.extend(prediction.tolist())

    pbar.close()
    toc = time.time()
    total_time = toc - tic
    test_report = classification_report(test_labels, test_preds, target_names=classes_name)
    test_matrix = confusion_matrix(test_labels, test_preds)
    print('\nTest result:',
          '\n Total loss:', total_loss.avg,
          '\n - Accuracy:', total_accu.avg,
          '\n - Total time:', total_time,
          '\n - Report: \n', test_report,
          '\n - Confussion Matrix: \n', test_matrix
          )


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_DATA_FOLDER, transform=transform)
    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, num_workers=N_CPU, shuffle=True, pin_memory=True)

    val_data = datasets.ImageFolder(VAL_DATA_FOLDER, transform=transform)
    val_dataloader = DataLoader(
        val_data, batch_size=BATCH_SIZE, num_workers=N_CPU, shuffle=True, pin_memory=True)

    # test_data = datasets.ImageFolder(TEST_DATA_FOLDER, transform=transform)
    # test_dataloader = DataLoader(
    #     test_data, batch_size=16, num_workers=4, shuffle=True, pin_memory=True)

    model = Network()
    params = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params.append(np.prod(p.size()))
    print(np.sum(params))
    # train(model, train_dataloader, val_dataloader, N_EPOCHS)
    # test(model, test_dataloader)
