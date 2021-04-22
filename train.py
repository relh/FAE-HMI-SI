import argparse
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from hmi import HMI_Dataset
from unet import UNet

p = argparse.ArgumentParser()
# Choose which magnetic field parameter to predict
p.add_argument('--target', default='field', type=str, help='target field/inclinaiton/azimuth/vlos_mag/eta_0/src_grad/src_continuum')
p.add_argument('--norotate', dest='norotate', action='store_true', help='whether to train unrotated model')

# Specify GPU to load network to and run image on
p.add_argument('--device', default='cuda:0', type=str, help='cuda GPU to run the network on')
args = p.parse_args()

# Load ZARRs containing data
x_labels = ['contin'] + (['meta'] if not args.norotate else []) + ['iquv']
train_dataset = HMI_Dataset('./HMIFull_ZARR/', x_labels=x_labels, y_labels=[args.target])
test_dataset = HMI_Dataset('./HMI2016_ZARR2/', x_labels=x_labels, y_labels=[args.target])

# Specify an arbitrary division of the data, first 3/5 training, next 1/5 validation.
# These ranges are sequentially assigned due to test data occuring subsequently. 
train_indices = range(int(len(train_dataset)*3/5))
val_indices = range(int(len(train_dataset)*3/5), int(len(train_dataset)*4/5))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, sampler=SubsetRandomSampler(train_indices), num_workers=1, pin_memory=False)
val_loader = DataLoader(train_dataset, batch_size=1, sampler=SubsetRandomSampler(val_indices), num_workers=1, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=False)

# Create model and initialize optimizers
net = UNet(25 if args.norotate else 28, 1, batchnorm=False, dropout=0.3, regression=False, bins=80, bc=64).to(args.device)
optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-3)#, betas=(0.5, 0.999))
rlrop = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training loop
bins = 80
epoch_len = 2500

def run_epoch(data_loader, net, optimizer, rlrop, epoch, is_train=True):
    start = time.time()
    losses = 0

    torch.set_grad_enabled(is_train)
    net = net.train() if is_train else net.eval()
    pbar = tqdm(data_loader, ncols=300)
    for i, batch in enumerate(pbar):
        # ================ preprocess in/output =====================
        im_inp = batch['X']
        field_mask = batch['field_mask']

        # =============== classification target =====================
        class_target = batch['Y']
        if 'vlos_mag' in args.target: class_target += 700000.0
        class_target = (class_target * ((bins-1) / train_dataset.max_divisor)).clamp(0, bins-1)
        mod_shape = list(class_target.shape) + [bins]

        class_target = class_target.unsqueeze(4)
        class_ones = torch.ones(class_target.shape, requires_grad=False)
        class_zeros = torch.zeros(mod_shape, requires_grad=False)
        class_zeros1 = torch.zeros(mod_shape, requires_grad=False)

        floored = class_target.floor()
        plus1 = (floored + 1).clamp(0, bins-1) # the next bin, but clamping to avoid oob
        class_target = class_target - floored       # how much weight you want

        # scatter both and multiply
        floored = (class_zeros.scatter_(4, floored.long(), class_ones) * (1.0-class_target)).view(-1, 1, bins)
        plus1 = (class_zeros1.scatter_(4, plus1.long(), class_ones) * class_target).view(-1, 1, bins)
        class_target = (floored + plus1) + 1e-4
        class_target = class_target.squeeze() / class_target.sum(dim=2)

        field_mask = field_mask.to(args.device).flatten()
        class_target = class_target.to(args.device)[field_mask > 0.7]

        # ================== forward + losses =======================
        optimizer.zero_grad()

        pred = net(im_inp.to(args.device))

        pred = torch.nn.functional.log_softmax(pred, dim=1)
        pred = pred.reshape(1, bins, -1).permute(0,2,1).reshape(-1, bins)[field_mask > 0.7]
        loss = torch.nn.KLDivLoss(reduction='sum')(pred, class_target)

        if is_train:
            loss.backward()
            optimizer.step()

        # ================== logging ====================
        losses += float(loss.detach())
        step = (i + epoch * epoch_len)

        pbar.set_description(
            '{} epoch {}: itr {:<6}/ {}- {}- iml {:.4f}- aiml {:.4f}- dt {:.4f}'
          .format('TRAIN' if is_train else 'VAL  ', 
                  epoch, i * data_loader.batch_size, len(data_loader) * data_loader.batch_size, # steps
                  args.target,
                  loss / data_loader.batch_size, losses / (i+1), # print batch loss and avg loss
                  time.time() - start)) # batch time

        # ================== termination ====================
        if i > (epoch_len / data_loader.batch_size): break

    avg_loss = losses / (i+1)
    if not is_train:
        rlrop.step(avg_loss)
    return avg_loss

# Meta training loop
train_losses, val_losses = [], []
min_loss = sys.maxsize
failed_epochs = 0

for epoch in range(100):
    train_losses.append(run_epoch(train_loader, net, optimizer, rlrop, epoch, is_train=True))
    val_losses.append(run_epoch(val_loader, net, optimizer, rlrop, epoch, is_train=False))

    if val_losses[-1] < min_loss:
        min_loss = val_losses[-1]
        failed_epochs = 0
        model_out_path = './models/' + '_'.join([args.target, str(epoch), str(float(min_loss))]) + '.pth'
        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_out_path)
        print('saved model..\t\t\t {}'.format(model_out_path))
    else:
        failed_epochs += 1
        print('--> loss failed to decrease {} epochs..\t\t\tthreshold is {}, {} all..{}'.format(failed_epochs, 6, val_losses, min_loss))
        if failed_epochs > 4: break
