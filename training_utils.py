import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

def print_model_config(total_params, batch_size, device, train_batches):
    print(f"\nModel Configuration:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Total training batches per epoch: {train_batches}")
    
    print("-" * 110)
    print(f"{'Epoch':^7} | {'Batch':^12} | {'Train Loss':^12} | {'Train Acc':^10} | {'Test Loss':^11} | "
          f"{'Test Acc':^9} | {'Diff':^8} | {'LR':^9}")
    print("-" * 110)

def print_epoch_stats(epoch, batch_idx, total_batches, train_loss, train_acc, test_loss, test_acc, lr):
    batch_info = f"{batch_idx+1}/{total_batches}"
    acc_diff = test_acc - train_acc
    print(f"{epoch+1:^7d} | {batch_info:^12} | {train_loss:^12.6f} | {train_acc:^10.2f} | "
          f"{test_loss:^11.6f} | {test_acc:^9.2f} | {acc_diff:^8.2f} | {lr:^9.7f}")

def print_training_summary(current_lr, train_accuracies, test_accuracies, max_epochs, total_batches):
    print("-" * 110)
    print(f"Training Summary:")
    print(f"Final Learning Rate: {current_lr}")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
    print(f"Best Train Accuracy: {max(train_accuracies):.2f}%")
    print(f"Lowest Train-Test Gap: {min([t-v for t,v in zip(test_accuracies, train_accuracies)]):.2f}%")
    print(f"Final Train-Test Gap: {test_accuracies[-1]-train_accuracies[-1]:.2f}%")
    print(f"Total Epochs: {max_epochs}")
    print(f"Total Batches Processed: {total_batches}")

def get_progress_bar(train_loader, epoch, max_epochs):
    return tqdm(enumerate(train_loader), 
               total=len(train_loader),
               desc=f'Epoch {epoch+1}/{max_epochs}',
               leave=False) 