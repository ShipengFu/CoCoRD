from __future__ import print_function

import cocord.loader as loader
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}
std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_cifar100_dataloaders(opt):
    """
    cifar 100
    """
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    # =================== base data augmentation for student in MoCo model, train ===================
    train_transform_base = transforms.Compose([  # === same as student baseline training
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # =================== base data augmentation, validation ===================
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=opt['data_folder'], download=True, train=True,
                                  transform=loader.CoCoRDTransform(base_transform=train_transform_base))

    train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True,
                              num_workers=opt['num_workers'], pin_memory=opt['pin_memory'])

    val_set = datasets.CIFAR100(root=opt['data_folder'], download=True, train=False, transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=opt['batch_size'], shuffle=False, num_workers=opt['num_workers'])

    return train_loader, val_loader
