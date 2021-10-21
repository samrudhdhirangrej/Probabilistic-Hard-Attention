#!/usr/bin/env python
# coding: utf-8

# In[1]:

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np
from model import Model
from dataloader import dataloader
from datetime import datetime
from train_test import train_test
import fire
import os

def main(dataset='svhn', datapath='./', lr=0.001, training_phase='first', ccebal=1, batch=64, batchv=64, T=7, logfolder='./log', epochs=1, pretrain_checkpoint=None):

    try:
        os.mkdir(logfolder)
    except:
        pass

    CUDA_LAUNCH_BLOCKING=1
    torch.backends.cudnn.benchmark = True

    # overwrite ccebal for first two training phases
    if training_phase=='first':
        ccebal=1
    elif training_phase=='second':
        ccebal=0
    
    nsfL, nf, nh, nz, classes, gz, imsz, train_loader, test_loader = dataloader(dataset, batch, batchv, datapath)
    
    device = torch.device('cuda')
    model = Model(dataset, T, nsfL, nf, nh, nz, classes, gz, imsz, ccebal, training_phase, pretrain_checkpoint).to(device)

    optimizerG = optim.Adam(list([p for p in model.parameters() if p.requires_grad == True]), lr = lr) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=25, threshold=0.01, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)
   
    logger = SummaryWriter(logfolder+'/log_' + datetime.now().isoformat(sep='-'))
    
    train_test(batch, batchv, T, device, model, optimizerG, logfolder, logger, train_loader, test_loader, scheduler, epochs)

if __name__ == '__main__':
    fire.Fire(main)
