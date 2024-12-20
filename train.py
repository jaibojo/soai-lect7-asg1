import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from model1 import MNISTClassifier
import torch.nn as nn
import torch.optim as optim
from training_utils import (print_model_config, print_epoch_stats, 
                          print_training_summary, get_progress_bar)

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

def train_model(model_name="Model1", max_epochs=15, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    # Create indices for the split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    # Split indices
    train_idx = indices[:50000]  # First 50000 for training
    val_idx = indices[50000:]    # Remaining 10000 for validation
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss and optimizer
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Print initial configuration
    total_params = count_parameters(model)
    print_model_config(model_name, total_params, batch_size, device, len(train_loader))
    
    # Training loop
    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = get_progress_bar(train_loader, epoch, max_epochs)
        
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch+1}/{max_epochs} [Batch {batch_idx+1}/{len(train_loader)}]')
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc, _ = evaluate_model(model, train_loader, device, criterion)
        val_acc, val_loss = evaluate_model(model, val_loader, device, criterion)
        test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        
        print_epoch_stats(model_name, epoch, batch_idx, len(train_loader), train_loss, 
                         train_acc, test_loss, test_acc, current_lr)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        val_accuracies.append(val_acc)
    
    print_training_summary(current_lr, train_accuracies, test_accuracies, 
                          max_epochs, max_epochs * len(train_loader))
    
    print("\nValidation Set Performance:")
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    return train_accuracies, test_accuracies, val_accuracies

if __name__ == "__main__":
    # For Model1
    train_model("Model1")
    
    # For Model2
    # train_model("Model2") 