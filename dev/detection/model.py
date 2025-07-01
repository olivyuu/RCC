import torch
import torch.nn as nn
import torchvision

def get_model(arch="densenet121", pretrained=False):
    model_fn = getattr(torchvision.models, arch)
    model = model_fn(pretrained=pretrained)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 1)
    )
    return model
