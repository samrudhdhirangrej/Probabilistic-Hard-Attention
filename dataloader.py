import torch
from torchvision import datasets, transforms
import numpy as np
import random

def dataloader(dataset, batch, batchv, datapath):


    if dataset=='svhn':

        kwargs = {'num_workers': 4, 'pin_memory': True}
        
        train_dataset = torch.utils.data.ConcatDataset([
                    datasets.SVHN(
                        root=datapath+'/svhn/data',
                        split='train',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                        download=True),
                    datasets.SVHN(
                        root=datapath+'/svhn/data',
                        split='extra',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                        download=True)])
                    
        train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch, shuffle=True, **kwargs)
        
        test_dataset = datasets.SVHN(
                    root=datapath+'/svhn/data',
                    split='test',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                    download=True)
        
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batchv, shuffle=False, **kwargs)

        nf = 128
        nz = 256
        nh = 512
        classes = 10
        imsz = 32
        gz = 8
        nsfL=4

    if dataset=='cifar10':

        kwargs = {'num_workers': 4, 'pin_memory': True}
        
        train_dataset = datasets.CIFAR10(
                root=datapath+'/cifar10/data',
                train=True,
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None,
                download=True)
        
        train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch, shuffle=True, **kwargs)
        
        test_dataset = datasets.CIFAR10(
                root=datapath+'/cifar10/data',
                train=False,
                transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None,
                download=True)
        
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batchv, shuffle=False, **kwargs)
 
        nf = 128
        nz = 256
        nh = 512
        classes = 10
        imsz = 32
        gz = 8
        nsfL=4

    elif dataset=='cifar100':

        kwargs = {'num_workers': 4, 'pin_memory': True}
        
        train_dataset = datasets.CIFAR100(
                root=datapath+'/cifar100/data',
                train=True,
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None,
                download=True)
        
        train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch, shuffle=True, **kwargs)
        
        test_dataset = datasets.CIFAR100(
                root=datapath+'/cifar100/data',
                train=False,
                transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None,
                download=True)
        
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batchv, shuffle=True, **kwargs)
 
        nf = 512
        nz = 1024
        nh = 2048
        classes = 100
        imsz = 32
        gz = 8
        nsfL=6

    elif dataset=='tinyimagenet':

        kwargs = {'num_workers': 4, 'pin_memory': True}

        train_dataset = datasets.ImageFolder(
                root=datapath+'/tinyimagenet/data/tiny-imagenet-200/train',
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(64, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None)
        
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch, shuffle=True, **kwargs)

        test_dataset = datasets.ImageFolder(
                root=datapath+'/tinyimagenet/data/tiny-imagenet-200/val',
                transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
                target_transform=None)
        
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batchv, shuffle=True, **kwargs)
 
        nf = 512
        nz = 1024
        nh = 2048
        classes = 200
        imsz = 64
        gz = 16
        nsfL=6

    elif dataset=='cinic10':

        kwargs = {'num_workers': 4, 'pin_memory': True}
       
        train_data, train_label = torch.load(datapath+'/cinic10/data/train_data.pth')
        valid_data, valid_label = torch.load(datapath+'/cinic10/data/valid_data.pth')
        test_data ,  test_label = torch.load(datapath+'/cinic10/data/test_data.pth')
        trainval_data = torch.cat([train_data, valid_data])
        trainval_label = torch.cat([train_label, valid_label])

        train_dataset = TransformedTensorDataset(
                trainval_data, trainval_label,
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
        
        train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch, shuffle=True, **kwargs)
        
        test_dataset = TransformedTensorDataset(
                test_data, test_label,
                transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batchv, shuffle=True, **kwargs)
 
        nf = 128
        nz = 256
        nh = 512
        classes = 10
        imsz = 32
        gz = 8
        nsfL=4

    return nsfL, nf, nh, nz, classes, gz, imsz, train_loader, test_loader
  

class TransformedTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index].float()/255.0
        if self.transform:
            x = self.transform(x)
        y = self.labels[index].long()
        return x, y

    def __len__(self):
        return self.labels.size(0)
       

