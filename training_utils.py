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

def print_model_config(model_name, total_params, batch_size, device, batches_per_epoch):
    print(f"\nModel: {model_name}")
    print(f"Total trainable parameters: {total_params}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Total training batches per epoch: {batches_per_epoch}\n")
    
    print("--------------------------------------------------------------------------------------------------------")
    print(" Model |  Epoch  |    Batch     |  Train Loss  | Train Acc  |  Test Loss  | Test Acc  |  Diff   |    LR    ")
    print("--------------------------------------------------------------------------------------------------------")

def print_epoch_stats(model_name, epoch, batch_idx, total_batches, train_loss, train_acc, 
                     test_loss, test_acc, current_lr):
    print(f" {model_name:5} |   {epoch+1:2d}    |   {batch_idx+1:4d}/{total_batches:<4d} |   {train_loss:.6f}   |   {train_acc:.2f}    |  {test_loss:.6f}   |   {test_acc:.2f}   |  {test_acc-train_acc:5.2f}  | {current_lr:.6f}")

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