import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier
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

def train_model(max_epochs=15, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss and optimizer
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Print initial configuration
    total_params = count_parameters(model)
    print_model_config(total_params, batch_size, device, len(train_loader))
    
    # Training loop
    train_accuracies = []
    test_accuracies = []
    
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
        test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        
        print_epoch_stats(epoch, batch_idx, len(train_loader), train_loss, 
                         train_acc, test_loss, test_acc, current_lr)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    print_training_summary(current_lr, train_accuracies, test_accuracies, 
                          max_epochs, max_epochs * len(train_loader))
    
    return train_accuracies, test_accuracies

if __name__ == "__main__":
    train_model() 