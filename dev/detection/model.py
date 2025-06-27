import torch
import torch.nn as nn
import torchvision

def get_model(pretrained=False):
    # Use DenseNet for patch classification
    model = torchvision.models.densenet121(pretrained=pretrained)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # single channel
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1)
    )
    return model
