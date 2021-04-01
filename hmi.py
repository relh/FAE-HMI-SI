import os
import pickle

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

zarr.blosc.set_nthreads(10)

class HMI_Dataset(Dataset):
    def __init__(self, data_path=None, x_labels=['contin', 'meta', 'iquv'], y_labels=['field'], bins=80):
        '''Raw SDO HMI Data. This dataset loads the inputs to the VFISV ME Inversion, 
        which are a series of I, Q, U, V polarized light at 6 bandpasses, as well as
        related metadata and continuum data.

        Inputs:
            meta: OFF_DISC {0,1}, SLONG (?,?), SLAT (?,?)
            continuum: HMI Continuum [0, inf]
            iquv: I0, I1, .., Q0, Q1, ... U0, U1, ..., V0, V1

        Outputs:
            0 - B aka field strength [0, \inf]
            1 - Incl in [0, 180]
            2 - Azimuth (for HMI, if it's been done CROTA) in [0, 180]
            5 - Vlos_Mag in [-700k, 700k]
            6 - Dop_Width in [0, 60.0]
            7 - Eta_0 in [0, 50.0]
            8 - Src_Continuum in [0, ~29,000]
            9 - Src_Grad in [0, ~53,000]
        '''
        self.data_path = data_path
        self.x_labels = x_labels 
        self.y_labels = y_labels 
        self.bins = bins 

        self.normalize = 'rzscore' # either rzscore or None for raw data
        self.max_divisor = None
        if 'field' in self.y_labels:           self.max_divisor = 5000.0
        elif 'inclination' in self.y_labels:   self.max_divisor = 180.0
        elif 'azimuth' in self.y_labels:       self.max_divisor = 180.0
        elif 'vlos_mag' in self.y_labels:      self.max_divisor = 700000.0 + 700000.0 # includes negative values
        elif 'dop_width' in self.y_labels:     self.max_divisor = 60.0
        elif 'eta_0' in self.y_labels:         self.max_divisor = 50.0
        elif 'src_continuum' in self.y_labels: self.max_divisor = 29060.61
        elif 'src_grad' in self.y_labels:      self.max_divisor = 52695.32

        # ZARRs contain data 
        self.meta = zarr.open(data_path + 'META.zarr', mode='r')
        self.contin = zarr.open(data_path + 'CONTIN.zarr', mode='r')
        self.iquv = zarr.open(data_path + 'IQUV.zarr', mode='r')
        self.y = zarr.open(data_path + 'Y.zarr', mode='r')
        self.image_sizes = np.load(data_path + 'imageSizes.npy')
        self.dates = open(data_path + 'dates_zarr.txt').readlines()

        self.x_normalize_params = None
        self.y_normalize_params = None
        self.set_normalize()

    def set_normalize(self, params=None):
        if params is None:
            params_path = './params/ondisk_params_{}_{}_{}.pkl'.format(str('-'.join(self.x_labels)), str('-'.join(self.y_labels)), str(self.bins))
            if os.path.exists(params_path):
                with open(params_path, 'rb') as handle:
                    params = pickle.load(handle)
            else:
                params = self.get_normalize()
                with open(params_path, 'wb') as handle:
                    pickle.dump(params, handle)

        self.x_normalize_params = (params['X_median'], params['X_iqr'])
        self.y_normalize_params = (params['Y_median'], params['Y_iqr'])
        self.params = params

    def get_normalize(self):
        normalize = self.normalize
        self.normalize = None

        all_X_pts = []
        all_Y_pts = []

        for i in [x for x in range(len(self))]:
            item = self[i]
            size = [int(x) for x in item['size']]
            print('building normalization parameters.. ' + str(i) + ":\t" + str(size) + " size")

            X = item['X'][:, :size[0], :size[1]]
            Y = item['Y'][:, :size[0], :size[1]]
            field = item['field_mask'][:, :size[0], :size[1]]

            # on-disk pixels are those with field greater than 0.7 Mx/cm^2
            over07 = (field > 0.7).nonzero()
            indices = np.random.choice(over07.shape[0], size=int(over07.shape[0] / 50))
            weight_x_indices = over07[:, 1][indices] 
            weight_y_indices = over07[:, 2][indices]

            all_X_pts.append(X[:, weight_x_indices, weight_y_indices])
            all_Y_pts.append(Y[:, weight_x_indices, weight_y_indices])

        print('concatenating..')
        XSamples = np.concatenate(all_X_pts, axis=1)
        YSamples = np.concatenate(all_Y_pts, axis=1)

        print('percentiles..')
        x_percentiles = np.percentile(XSamples,[1,25,75,99],axis=1)
        y_percentiles = np.percentile(YSamples,[0.01, 0.1, 1, 5, 25, 75, 95, 99, 99.9, 99.99],axis=1)

        print('calculated everything')
        self.normalize = normalize
        return {'X_mean': np.mean(XSamples,axis=1),
                'X_median': np.median(XSamples,axis=1),
                'X_iqr': np.squeeze(x_percentiles[2,:]-x_percentiles[1,:]), 
                \
                'Y_mean': np.mean(YSamples,axis=1),
                'Y_median': np.median(YSamples,axis=1),
                'Y_iqr': np.squeeze(y_percentiles[5,:]-y_percentiles[4,:]), 
                \
                'max_divisor': self.max_divisor, 
                }

    def normalize_item(self,X,params,y=False):
        if self.normalize is None:
            return X 
        else:
            med, scale = params[0], np.atleast_1d(1.5*(params[1]+1e-1))
            XScored = (X - med[:,np.newaxis,np.newaxis]) / (scale[:,np.newaxis,np.newaxis])
            return np.clip(XScored,-50,50)

    def denormalize_item(self,X,params):
        if self.normalize is None:
            return X 
        else:
            med, scale = params[0], 1.5*(params[1]+1e-1)
            return (X*scale)+med

    def fetch_item(self, zarr, idx, y_index=None):
        item = zarr[idx] 
        item[item != item] = 0.0 # remove nans
        if y_index is not None:
            item = item[y_index, :, :]
        item = torch.from_numpy(item.astype(np.float32))
        return item

    def __len__(self):
        return len(self.dates)

    def __getitem__(self,idx):
        X_tensors = []
        Y_tensors = []

        date = self.dates[idx]
        field_mask = self.fetch_item(self.y, idx, y_index=0)

        if 'contin' in self.x_labels: X_tensors.append(self.fetch_item(self.contin, idx))
        if 'meta' in self.x_labels:   X_tensors.append(self.fetch_item(self.meta, idx))
        if 'iquv' in self.x_labels:   X_tensors.append(self.fetch_item(self.iquv, idx))

        if 'field' in self.y_labels:         Y_tensors.append(self.fetch_item(self.y, idx, y_index=0))
        if 'inclination' in self.y_labels:   Y_tensors.append(self.fetch_item(self.y, idx, y_index=1))
        if 'azimuth' in self.y_labels:       Y_tensors.append(self.fetch_item(self.y, idx, y_index=2))
        if 'vlos_mag' in self.y_labels:      Y_tensors.append(self.fetch_item(self.y, idx, y_index=5))
        if 'dop_width' in self.y_labels:     Y_tensors.append(self.fetch_item(self.y, idx, y_index=6))
        if 'eta_0' in self.y_labels:         Y_tensors.append(self.fetch_item(self.y, idx, y_index=7))
        if 'src_continuum' in self.y_labels: Y_tensors.append(self.fetch_item(self.y, idx, y_index=8))
        if 'src_grad' in self.y_labels:      Y_tensors.append(self.fetch_item(self.y, idx, y_index=9))
        size = self.image_sizes[idx]

        return {'index': idx,
                'date': date,
                'size': size,
                'X_labels': self.x_labels,
                'X': self.normalize_item(torch.cat(X_tensors), self.x_normalize_params).float(),
                'Y_labels': self.y_labels,
                'Y': torch.cat(Y_tensors).unsqueeze(0).float(),
                'Y_norm': self.normalize_item(torch.cat(Y_tensors), self.y_normalize_params, y=True).float(),
                'field_mask': field_mask.unsqueeze(0).float()}
