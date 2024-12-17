import pytest
import torch
from model import MNISTClassifier
from train import train_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTClassifier()
    param_count = count_parameters(model)
    assert param_count < 8000, f"Model has {param_count} parameters, should be less than 8000"

def test_epoch_count():
    accuracies = train_model(max_epochs=15)
    assert len(accuracies) <= 15, "Training should not exceed 15 epochs"

def test_accuracy_threshold():
    accuracies = train_model(max_epochs=15)
    high_accuracy_count = sum(1 for acc in accuracies if acc >= 99.4)
    assert high_accuracy_count >= 3, "Model should achieve 99.4% accuracy in at least 3 epochs"

if __name__ == "__main__":
    pytest.main([__file__]) 