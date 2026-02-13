"""
Train DECENT-plus model: maternal/embryo (e.g. cumulus+PB vs TE+ICM) from single-cell methylation reads.
Run from project root: PYTHONPATH=. python -m stage1_decent.train --reads_dir <dir> --save_dir <dir>
Or from stage1_decent: python train.py --reads_dir <dir> --save_dir <dir>
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from .data_loader import data_prepare
    from .model import DISMIR_deep
except ImportError:
    from data_loader import data_prepare
    from model import DISMIR_deep


class Mydataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def validate_model(model, test_loader, criterion, threshold=0.5, device="cuda"):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    y_true, y_score = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs.permute(0, 2, 1))
            y_true.append(labels.cpu().numpy())
            y_score.append(outputs.cpu().numpy())
            if outputs.size(1) == 1:
                loss = criterion(outputs[:, 0], labels)
                predicted = (outputs > threshold).float()
            else:
                loss = criterion(outputs, labels.long())
                _, predicted = torch.max(outputs.data, 1)
            val_loss += loss.item()
            correct_val += (predicted.flatten() == labels).sum().item()
            total_val += labels.size(0)
    val_acc = correct_val / total_val
    val_loss = val_loss / len(test_loader)
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    return val_loss, val_acc, y_true, y_score


def train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler,
    max_lr, save_dir, num_epochs=10, log_per=100, save_per=10, threshold=0.5, device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, f'L{max_lr}_models')
    log_dir = os.path.join(save_dir, f'L{max_lr}_logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    total_step = len(train_loader) * num_epochs
    log_interval = max(1, total_step // log_per)
    save_interval = max(1, total_step // save_per)
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        correct_pred, total_pred, train_loss = 0, 0, 0.0
        for iteration, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs.permute(0, 2, 1))
            optimizer.zero_grad()
            if outputs.size(1) == 1:
                loss = criterion(outputs[:, 0], labels)
            else:
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if outputs.size(1) == 1:
                predicted = (outputs > threshold).float()
            else:
                _, predicted = torch.max(outputs.data, 1)
            correct_pred += (predicted.flatten() == labels).sum().item()
            total_pred += labels.size(0)

            if (iteration + 1) % log_interval == 0:
                avg_loss = train_loss / log_interval
                avg_acc = correct_pred / total_pred
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{iteration+1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                writer.add_scalar('Training/Loss', avg_loss, epoch * len(train_loader) + iteration)
                writer.add_scalar('Training/Accuracy', avg_acc, epoch * len(train_loader) + iteration)
                train_loss, correct_pred, total_pred = 0.0, 0, 0
            if (iteration + 1) % save_interval == 0:
                path = os.path.join(model_dir, f'model_epoch_{epoch+1}_step_{iteration+1}.pth')
                torch.save(model.state_dict(), path)

        scheduler.step()
        val_loss, val_acc, y_true, y_score = validate_model(model, test_loader, criterion, threshold, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))
        pd.DataFrame(np.hstack([y_true.reshape(-1, 1), y_score])).to_csv(
            os.path.join(model_dir, f'eval_res_{epoch+1}.csv')
        )
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reads_dir', required=True, help='Directory of .reads files (per cell type suffix)')
    parser.add_argument('--save_dir', required=True, help='Directory for checkpoints and logs')
    parser.add_argument('--datasize', type=int, default=7500000, help='Max reads per class')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    cell_type_suffix_dict = {'cumulus': '_G', 'PB': '_PB', 'te': 'TE_', 'icm': '_I'}
    cell_type_dict = {'cumulus': 0, 'PB': 0, 'te': 1, 'icm': 1}
    n_classes = len(set(cell_type_dict.values()))

    data, label = data_prepare(
        cell_type_dict, cell_type_suffix_dict,
        reads_dir=args.reads_dir, datasize=args.datasize
    )
    print('Data size:', len(data), 'Label mean:', float(np.mean(label)))

    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=0.2, random_state=42, stratify=label
    )
    del data, label
    gc.collect()

    train_dataset = Mydataset(train_data, train_label)
    test_dataset = Mydataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DISMIR_deep(n_classes).to(device)
    lr = 0.01
    criterion = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        lr, args.save_dir, num_epochs=args.epochs, device=device
    )


if __name__ == '__main__':
    main()
