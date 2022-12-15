"""
# File       : train.py
# Time       ：2022/11/23 12:39
# Author     ：Peng Cheng
# Description：
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import csv
import math
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import random
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from dataset import *
import wandb
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# Set your trainer.
def train(config,model):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer.lr, momentum=config.optimizer.momentum)
    # optimizer = torch.optim.Adam(model.parameters(),lr=config.optimizer.lr,weight_decay=config.optimizer.weight_decay)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.schedular.T_max, eta_min=config.schedular.eta_min, last_epoch=-1, verbose=False)


    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)  # Create directory of saving models.
    n_epochs, best_loss,best_acc, step, early_stop_count = config.epoch, math.inf, 0, 0, 0

    train_dataloader = get_SeaAnimal_dataloader(config,'train')
    valid_dataloader = get_SeaAnimal_dataloader(config,'valid')
    print("dataloader down")
    model.to(torch.device(config.device))

    for epoch in range(n_epochs):
        # =========================train============================
        model.train()  # Set your model to train mode.
        loss_record = []
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_dataloader, position=0, leave=True)
        train_correct = 0
        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(torch.float32).to(torch.device(config.device)), y.to(torch.float32).to(torch.device(config.device))  # Move your data to config.device.
            pred = model(x)
            loss = criterion(pred, y.long())  # the second parameter must be long
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            scheduler_lr.step() # Update lr
            step += 1
            loss_record.append(loss.detach().item())

            predict = pred.data.max(1, keepdim=True)[1]
            train_correct += predict.eq(y.data.view_as(predict)).sum()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        wandb.log({"train_loss": mean_train_loss})
        train_acc = train_correct / len(train_dataloader.dataset)
        wandb.log({"train_acc": train_acc})

        # =========================valid============================
        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        valid_correct = 0
        for x, y in valid_dataloader:
            x, y = x.to(torch.float32).to(config.device), y.to(torch.float32).to(config.device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y.long())
                predict = pred.data.max(1, keepdim=True)[1]
                valid_correct += predict.eq(y.data.view_as(predict)).sum()

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        wandb.log({"val_loss": mean_valid_loss})
        valid_acc = valid_correct / len(valid_dataloader.dataset)
        wandb.log({"val_acc": valid_acc})

        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}',
              'Train Accuracy: {}/{} ({:.0f}%)'.format(train_correct, len(train_dataloader.dataset),100. * train_correct / len(train_dataloader.dataset)),
              'Valid Accuracy: {}/{} ({:.0f}%)'.format(valid_correct, len(valid_dataloader.dataset),100. * valid_correct / len(valid_dataloader.dataset)))

        # ==================early stopping======================
        if os.path.exists(config.save_dir):
            pass
        else:
            os.mkdir(config.save_dir)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), config.save_dir + f'/epoch{epoch}.pt')  # Save your best model
            print('Saving model with acc{:.0f}%'.format(100.*best_acc))
            early_stop_count = 0
        else:
            early_stop_count += 1



def test(config, model):
    test_dataloader = get_SeaAnimal_dataloader(config,'test')
    model.to(torch.device(config.device))
    model.eval()  # Set your model to evaluation mode.
    correct = 0
    test_loss = 0
    test_losses = []
    test_label = []
    test_pred = []
    for x, y in tqdm(test_dataloader):
        x, y = x.to(torch.float32).to(torch.device(config.device)), y.to(torch.float32).to(torch.device(config.device))
        with torch.no_grad():
            out = model(x)
            test_loss += F.cross_entropy(out, y.long())
            pred = out.data.max(1, keepdim=True)[1]
            test_label += np.array(y.cpu()).astype(int).tolist()
            test_pred += np.array(pred.reshape([-1]).cpu()).tolist()
            correct += pred.eq(y.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

    # add more evaluation
    test_label = np.array(test_label)
    test_pred = np.array(test_pred)

    Precision = precision_score(test_label, test_pred, labels=[0, 1, 2], average='macro')
    Recall = recall_score(test_label, test_pred, labels=[0, 1, 2], average='macro')
    F1 = f1_score(test_label, test_pred, labels=[0, 1, 2], average='macro')
    confusion = confusion_matrix(test_label, test_pred)  # 混淆矩阵
    report = classification_report(test_label, test_pred)  # 每一类明细



    print("Precision_macro:  {}".format(Precision))
    print("Recall_macro:  {}".format(Recall))
    print("F1_macro:  {}".format(F1))
    print('Confusion Matrix :\n', confusion)
    print("Report:\n", report)


    classes = ['Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus',
     'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks',
     'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale']

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 11))
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap="Blues",
        ax=ax,  # 同上
        xticks_rotation="vertical",  # 同上
        values_format="d",  # 显示的数值格式
    )

    plt.savefig(config.save_dir+'.png')
    # plt.show()




