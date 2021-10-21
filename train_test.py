#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

def train_test(batch, batchv, T, device, netG, optimizerG, PATH, logger, train_loader, test_loader, scheduler, epochs):
    for epoch in range(epochs):
        logger.add_scalar('learning_rate/epoch_lr', optimizerG.param_groups[-1]['lr'], epoch)
        train_loss = 0
        train_acc = 0
        netG.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            loss, acc = netG(data, label)   
            train_loss += (loss.item()*data.size(0))
            train_acc += (acc[-1].item()*data.size(0))
            optimizerG.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=10)
            optimizerG.step()
    
            if (batch_idx%100)==0:
                with open(PATH+'/print.txt','a+') as f:
                    f.write('epoch {} batch {}/{} loss flow {:.2f} acc {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n'.format(epoch, batch_idx, len(train_loader), loss.item(), *acc))

            if batch_idx==10:
               break

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
    
        with open(PATH+'/print.txt','a+') as f:
            f.write('final train loss {:.2f} acc {:.2f}\n'.format(train_loss, train_acc))
               
        torch.save([netG.state_dict(), optimizerG.state_dict()], PATH+'/weights_f_{0}.pth'.format(epoch))
        np.savez(PATH+'/train_loss_acc_{0}.pth'.format(epoch), loss = train_loss, acc = train_acc)
        logger.add_scalar('loss/epoch_train', train_loss, epoch)
        logger.add_scalar('acc/epoch_train_{}'.format(T-1), train_acc, epoch)
        scheduler.step(train_loss)
    
        test_acc = [0 for _ in range(T)]
        netG.eval()
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                _, acc = netG.sample(data, label)    
            
            for i in range(T):
                test_acc[i] += (acc[i].item() * data.size(0))
        
            if batch_idx%100==0:
                with open(PATH+'/print.txt','a+') as f:
                    f.write('test epoch {} batch {}/{} acc {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n'.format(epoch, batch_idx, len(test_loader), *acc))

            if batch_idx==10:
               break

        for i in range(T):
            test_acc[i] /= len(test_loader.dataset)
            logger.add_scalar('acc/epoch_test_{}'.format(i), test_acc[i], epoch)
        
            with open(PATH+'/print.txt','a+') as f:
                f.write('final test acc {:.2f}\n'.format(test_acc[i]))
               
        np.savez(PATH+'/test_acc_{0}.npz'.format(epoch), acc = test_acc)
            

