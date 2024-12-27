import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from model5 import MNISTClassifier
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

def get_image_difficulties(model, dataset, indices, device, batch_size=128):
    """Calculate difficulty (loss) for each image in the given indices"""
    model.eval()
    difficulties = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Create a loader that only loads the specified indices
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            difficulties.extend(losses.cpu().numpy().tolist())
    
    difficulties = np.array(difficulties)
    if np.any(np.isnan(difficulties)):
        print("Warning: Found NaN difficulties, replacing with max value")
        max_val = np.nanmax(difficulties)
        difficulties = np.nan_to_num(difficulties, nan=max_val)
    
    print(f"\nDifficulty stats for {len(indices)} images:")
    print(f"Mean loss: {np.mean(difficulties):.4f}")
    print(f"Max loss: {np.max(difficulties):.4f}")
    print(f"Min loss: {np.min(difficulties):.4f}")
    
    return difficulties

def create_sorted_sampler(difficulties, indices):
    """Create sampler that returns indices sorted by difficulty"""
    # Pair difficulties with their indices and sort
    diff_idx_pairs = list(zip(difficulties, indices))
    sorted_pairs = sorted(diff_idx_pairs, key=lambda x: x[0], reverse=True)  # Higher loss = more difficult
    
    # Extract sorted indices
    sorted_indices = [idx for _, idx in sorted_pairs]
    return SubsetRandomSampler(sorted_indices)

def create_split_random_dataset(dataset, sorted_indices, batch_size=128):
    """Create a dataset that mixes random hard and easy images in each batch"""
    class SplitRandomDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, sorted_indices, batch_size):
            self.dataset = original_dataset
            self.batch_size = batch_size
            self.half_batch = batch_size // 2
            
            # Split indices into harder and easier halves
            mid_point = len(sorted_indices) // 2
            self.hard_indices = sorted_indices[:mid_point]  # First 25000 (harder)
            self.easy_indices = sorted_indices[mid_point:]  # Last 25000 (easier)
            
            # Debug: Print first few indices from each half
            print("\nDebug - First 10 hard indices:", self.hard_indices[:10])
            print("Debug - First 10 easy indices:", self.easy_indices[:10])
            
            # Calculate number of complete batches possible
            self.num_batches = len(sorted_indices) // batch_size
            self.total_samples = self.num_batches * batch_size
            
            # Create mixed batches
            self.mixed_indices = []
            
            # Debug: Print first few batches composition
            print("\nDebug - First 3 batches composition:")
            for batch_num in range(min(3, self.num_batches)):
                # Randomly sample from harder half
                hard_samples = np.random.choice(self.hard_indices, size=self.half_batch, replace=False)
                # Randomly sample from easier half
                easy_samples = np.random.choice(self.easy_indices, size=self.half_batch, replace=False)
                
                print(f"\nBatch {batch_num + 1}:")
                print("Hard samples indices:", hard_samples[:5], "...")
                print("Easy samples indices:", easy_samples[:5], "...")
                
                # Combine and shuffle
                batch_indices = list(hard_samples) + list(easy_samples)
                np.random.shuffle(batch_indices)
                self.mixed_indices.extend(batch_indices)
            
            # Continue creating remaining batches
            for _ in range(3, self.num_batches):
                hard_samples = np.random.choice(self.hard_indices, size=self.half_batch, replace=False)
                easy_samples = np.random.choice(self.easy_indices, size=self.half_batch, replace=False)
                batch_indices = list(hard_samples) + list(easy_samples)
                np.random.shuffle(batch_indices)
                self.mixed_indices.extend(batch_indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.mixed_indices[idx]]
        
        def __len__(self):
            return len(self.mixed_indices)

    return SplitRandomDataset(dataset, sorted_indices, batch_size)

def create_hardest_first_dataset(dataset, sorted_indices, batch_size=128):
    """Create a dataset that presents hardest images first"""
    class HardestFirstDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, sorted_indices):
            self.dataset = original_dataset
            self.sorted_indices = sorted_indices  # Already sorted by difficulty (hardest first)
            
            # Debug: Print first few indices to verify sorting
            print("\nDebug - First 20 indices (hardest images):", self.sorted_indices[:20])
            print("Debug - Last 20 indices (easiest images):", self.sorted_indices[-20:])
        
        def __getitem__(self, idx):
            return self.dataset[self.sorted_indices[idx]]
        
        def __len__(self):
            return len(self.sorted_indices)

    return HardestFirstDataset(dataset, sorted_indices)

def train_model(model_name="Model1", max_epochs=15, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=1),    # Random rotation ±7°
        transforms.RandomAffine(
            degrees=0,                                     # No additional rotation
            translate=(0.1, 0.1),                         # Random shift up to 10%
            fill=1
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform)
    
    # Create indices for the split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    # Split indices
    train_idx = indices[:50000]  # First 50000 for training
    val_idx = indices[50000:]    # Remaining 10000 for validation
    
    # Create initial random samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss and optimizer
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Define learning rate phases
    phase1_stages = [  # Epoch 1
        {
            'portion': 1.0,
            'lr': 0.01,
            'momentum': 0.9
        }
    ]
    
    phase2_stages = [  # Epochs 2-5
        {
            'portion': 0.20,
            'lr': 0.008,
            'momentum': 0.90
        },
        {
            'portion': 0.20,
            'lr': 0.008,
            'momentum': 0.90
        },
        {
            'portion': 0.20,
            'lr': 0.008,
            'momentum': 0.90
        },
        {
            'portion': 0.20,
            'lr': 0.008,
            'momentum': 0.90
        },
        {
            'portion': 0.20,
            'lr': 0.008,
            'momentum': 0.90
        }
    ]
    
    phase3_stages = [  # Epochs 6-10
        {
            'portion': 0.20,
            'lr': 0.004,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.004,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.004,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.004,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.004,
            'momentum': 0.95
        }
    ]
    
    phase4_stages = [  # Epochs 11-15
        {
            'portion': 0.20,
            'lr': 0.001,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.001,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.001,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.001,
            'momentum': 0.95
        },
        {
            'portion': 0.20,
            'lr': 0.001,
            'momentum': 0.95
        }
    ]

    optimizer = optim.SGD(model.parameters(), 
                         lr=phase1_stages[0]['lr'],
                         momentum=phase1_stages[0]['momentum'],
                         nesterov=True)

    # Training loop
    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    prev_test_acc = None

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        # Choose stages based on epoch
        if epoch == 0:
            current_stages = phase1_stages
        elif 1 <= epoch <= 4:
            current_stages = phase2_stages
        elif 5 <= epoch <= 9:
            current_stages = phase3_stages
        else:  # epoch >= 10
            current_stages = phase4_stages
        
        # Sort by difficulty after each epoch (starting from epoch 1)
        if epoch >= 1:
            print(f"\nSorting images by difficulty for epoch {epoch+1}...")
            # Get difficulties only for training images
            difficulties = get_image_difficulties(model, full_train_dataset, train_idx, device)
            
            # Sort training indices by difficulty (hardest first)
            sorted_indices = np.argsort(-difficulties)  # Negative for descending order
            sorted_idx = np.array(train_idx)[sorted_indices]
            
            # Create hardest-first dataset and loader
            hardest_first_dataset = create_hardest_first_dataset(full_train_dataset, sorted_idx)
            train_loader = DataLoader(hardest_first_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False)
            
            print(f"Created batches in order of difficulty (hardest first)")

        # Print headers
        print("\nEpoch Progress:")
        print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")
        print("   Model   |  Epoch  | Stage  |    Batch    |   TR_LS    |  TR_AC   |   TE_LS   |  TE_AC  | TE-TR    |    LR   | Test Δ")
        print("-----------|---------|--------|-------------|------------|----------|-----------|---------|----------|---------|--------")
        
        # Track metrics for each stage
        for stage_idx, stage in enumerate(current_stages):
            stage_start = int(len(train_loader) * sum(s['portion'] for s in current_stages[:stage_idx]))
            stage_end = int(len(train_loader) * sum(s['portion'] for s in current_stages[:stage_idx+1]))
            
            # Update optimizer for this stage
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage['lr']
                param_group['momentum'] = stage['momentum']
            
            stage_running_loss = 0.0
            stage_batches = 0
            
            # Use enumerate to iterate through the loader properly
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx < stage_start:
                    continue
                if batch_idx >= stage_end:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                stage_running_loss += loss.item()
                stage_batches += 1
            
            # Calculate and print stage metrics
            if stage_batches > 0:
                stage_loss = stage_running_loss / stage_batches
                print(f"\nStage {stage_idx+1} completed:")
                print(f"Batches processed: {stage_batches}")
                print(f"Average loss: {stage_loss:.4f}")
                
                train_acc, _ = evaluate_model(model, train_loader, device, criterion)
                test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
                current_lr = optimizer.param_groups[0]['lr']
                
                print_epoch_stats(model_name, epoch, stage_idx+1, batch_idx, len(train_loader),
                                stage_loss, train_acc, test_loss, test_acc, current_lr, prev_test_acc)
                
                prev_test_acc = test_acc
        
        # Calculate epoch-end metrics
        train_loss = running_loss / len(train_loader)
        train_acc, _ = evaluate_model(model, train_loader, device, criterion)
        test_acc, test_loss = evaluate_model(model, test_loader, device, criterion)
        val_acc, val_loss = evaluate_model(model, val_loader, device, criterion)
        
        # Print epoch stats
        print_epoch_stats(model_name, epoch, 0, len(train_loader)-1, len(train_loader), 
                         train_loss, train_acc, test_loss, test_acc, current_lr, prev_test_acc)
        
        # Store metrics
        prev_test_acc = test_acc
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