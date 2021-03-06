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
p.add_argument('--file', default='./inputs/input_iquv.pkl', type=str, help='IQUV input file')
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
print('initialized UNet')

# Load either normal model, unrotated model, or model from saved weights
if args.norotate: saved_network_state = torch.load(f'./models_norotate/{args.target}_model_norotate.pth')
else:             saved_network_state = torch.load(f'./models/{args.target}_model.pth')
net.load_state_dict(saved_network_state['model'])
print(f'loaded model... {args.target}')
net.eval()
x_labels = ['contin'] + (['meta'] if not args.norotate else []) + ['iquv']

# Specify target maximum divisor
if 'field' in args.target:           max_divisor = 5000.0
elif 'inclination' in args.target:   max_divisor = 180.0
elif 'azimuth' in args.target:       max_divisor = 180.0
elif 'vlos_mag' in args.target:      max_divisor = 700000.0 + 700000.0 # includes negative values
elif 'dop_width' in args.target:     max_divisor = 60.0
elif 'eta_0' in args.target:         max_divisor = 50.0
elif 'src_continuum' in args.target: max_divisor = 29060.61
elif 'src_grad' in args.target:      max_divisor = 52695.32
print(f'setting max divisor... {max_divisor}')

# Use an input file as default, otherwise use an index in the ZARR if provided
if args.index == -1:
    input_data = torch.load(args.file)

    if args.norotate:
        input_data = torch.cat((input_data[:1], input_data[4:]))
        print(f'using no rotate... input data is {input_data.shape}')
    else:
        print(f'using full model... input data is {input_data.shape}')

    input_tiles = full_disk_to_tiles(input_data)
else:
    test_dataset = HMI_Dataset('./HMI2015_NoRotate_ZARR/' if args.norotate else './HMI2016_ZARR/', x_labels=x_labels, y_labels=[args.target], bins=80)

    input_tiles = []
    for i in range(16):
        input_tiles.append(test_dataset[args.index*16+i]['X'].unsqueeze(0))

# Run network on all 16 tiles
outputs = []
for i, tile in enumerate(input_tiles):
    print(f'running network... tile {i}')
    pred = net(tile.unsqueeze(0).to(args.device))
    pred = bins_to_output(pred, max_divisor)
    outputs.append(pred)

# Save the full disk after running across the whole thing.
print(f'saving output... ./outputs/{args.target}_{args.index}.png')
output = tiles_to_full_disk(outputs).cpu().numpy()

with open(f'./outputs/{args.target}_{args.index}.npy', 'wb') as f:
    np.save(f, output)

plt.imsave(f'./outputs/{args.target}_{args.index}.png', output)
