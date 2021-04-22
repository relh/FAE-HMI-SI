import argparse
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch

from hmi import HMI_Dataset
from unet import UNet
from util import bins_to_output, full_disk_to_tiles, tiles_to_full_disk

p = argparse.ArgumentParser()

# Specify either a file (default) or an index in the test ZARR to load input
p.add_argument('--file', default='./inputs/input_iquv.npy', type=str, help='IQUV input file')
p.add_argument('--index', default=-1, type=int, help='ZARR IQUV input index')

# Choose which magnetic field parameter to predict
p.add_argument('--target', default='field', type=str, help='target field/inclinaiton/azimuth/vlos_mag/eta_0/src_grad/src_continuum')
p.add_argument('--norotate', dest='norotate', action='store_true', help='whether to load unrotated model')

# Specify GPU to load network to and run image on
p.add_argument('--device', default='cuda:0', type=str, help='cuda GPU to run the network on')
args = p.parse_args()

# Initialize network 
torch.set_grad_enabled(False)
net = UNet(25 if args.norotate else 28, 1, batchnorm=False, dropout=0.3, regression=False, bins=80, bc=64).to(args.device)

# Load either normal model, unrotated model, or model from saved weights
if args.norotate: saved_network_state = torch.load(f'./models_norotate/{args.target}_model_norotate.pth')
else:             saved_network_state = torch.load(f'./models/{args.target}_model.pth')
net.load_state_dict(saved_network_state['model'])
net.eval()

x_labels = ['contin'] + (['meta'] if not args.norotate else []) + ['iquv']

# Use an input file as default, otherwise use an index in the ZARR if provided
if args.index == -1:
    input_data = torch.load(args.file)
    print(input_data.shape)

    if args.norotate:
        input_data = torch.cat(input_data[0], input_data[4:])
        print(input_data.shape)

    input_tiles = full_disk_to_tiles(input_data)
else:
    train_dataset = HMI_Dataset('./inputs/HMIFull_ZARR/', x_labels=x_labels, y_labels=[args.target], bins=80)
    test_dataset = HMI_Dataset('./inputs/HMI2016_ZARR2/', x_labels=x_labels, y_labels=[args.target], bins=80)

    input_tiles = []
    for i in range(16):
        input_tiles.append(test_dataset[args.index*16+i]['X'].unsqueeze(0))

# Run network on all 16 tiles
outputs = []
for tile in input_tiles:
    pred = net(tile.to(args.device))
    pred = bins_to_output(pred, test_dataset.max_divisor)
    outputs.append(pred)

# Save the full disk after running across the whole thing.
output = tiles_to_full_disk(outputs).cpu().numpy()

with open(f'./outputs/{args.target}_{args.index}.npy', 'wb') as f:
    np.save(f, output)

plt.imsave(f'./outputs/{args.target}_{args.index}.png', output)
