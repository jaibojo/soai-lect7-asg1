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

def print_model_config(model_name, total_params, batch_size, device, num_batches):
    print("\nModel Configuration:")
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"Batches per epoch: {num_batches}")
    print("\nTraining Progress:")
    # Header row 1 - Main columns
    print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")
    print("   Model   |  Epoch  | Stage  |    Batch    |   TR_LS    |  TR_AC   |   TE_LS   |  TE_AC  | TE-TR    |    LR   | Test Δ")
    print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")

def print_epoch_stats(model_name, epoch, stage, batch_idx, num_batches, train_loss, train_acc, test_loss, test_acc, lr, prev_test_acc=None):
    # Calculate test accuracy change
    test_acc_change = "NA" if prev_test_acc is None else f"{test_acc - prev_test_acc:+.2f}"
    
    # Convert float values to int for formatting
    epoch_num = int(epoch)
    stage_num = int(stage) if isinstance(stage, (int, float)) else stage
    batch_num = int(batch_idx) if isinstance(batch_idx, (int, float)) else batch_idx
    num_batch = int(num_batches) if isinstance(num_batches, (int, float)) else num_batches
    
    print(f" {model_name:^9} | {epoch_num+1:^7d} | {stage_num:^6d} | {batch_num+1:4d}/{num_batch:<4d} | "
          f"{train_loss:^10.6f} | {train_acc:^8.2f} | {test_loss:^9.6f} | {test_acc:^7.2f} | "
          f"{test_acc-train_acc:^8.2f} | {lr:^7.6f} | {test_acc_change:^6}")

def print_training_summary(current_lr, train_accuracies, test_accuracies, max_epochs, total_batches):
    print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")
    print("Training Summary:")
    print(f"Final LR: {current_lr:.6f}")
    print(f"Best TE_AC: {max(test_accuracies):.2f}%")
    print(f"Best TR_AC: {max(train_accuracies):.2f}%")
    print(f"Best TE-TR: {max([te-tr for te,tr in zip(test_accuracies, train_accuracies)]):.2f}%")
    print(f"Final TE-TR: {test_accuracies[-1]-train_accuracies[-1]:.2f}%")
    print(f"Total Epochs: {max_epochs}")
    print(f"Total Batches: {total_batches}")
    print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")

def get_progress_bar(train_loader, epoch, max_epochs):
    return tqdm(enumerate(train_loader), 
               total=len(train_loader),
               desc=f'Epoch {epoch+1}/{max_epochs}',
               leave=False) 