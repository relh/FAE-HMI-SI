# FAE-HMI-SI
## Fast and Accurate Emulation of the SDO/HMI Stokes Inversion with Uncertainty Quantification

Welcome to FAE-HMI-SI! 

This is code for the paper [Fast and Accurate Emulation of the SDO/HMI Stokes Inversion with Uncertainty Quantification](http://arxiv.org), 
accepted for publication in the [Astrophysical Journal](https://iopscience.iop.org/journal/0004-637X).

We train a UNet to map a 28-dimensional signal of polarized light (4 polarizations at 6 bandpasses + metadata) into a magnetic field vector,
representing the strength and direction of the magnetic field on the surface of the sun.

**To learn more, checkout the [interactive project website](https://relh.github.io/FAE-HMI-SI/):**

<p align="center">
<img src="./website/assets/website_screengrab_2.png" alt="project website screengrab" width="600"/>
</p>

## Installation 

Install [pytorch](pytorch.org) for use on a GPU, then install the remaining requirements:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Data Preparation 

To use pre-trained models, first download them (1.2 GB) from dropbox:

```
wget https://www.dropbox.com/s/x4lrx2npy4zv403/models.zip
unzip models.zip
```

Once you have models downloaded/trained, fetch an example `(28, 4096, 4096)` input tensor (1.8 GB) and run it:

```
wget https://www.dropbox.com/s/3itmkqcal4u0otl/inputs.zip
unzip inputs.zip
```

Alternatively, you can download ZARRs that contain the full year-long dataset (250+ GB each)

`COMING SOON`

## Inference

1. Run `python inference.py`.
2. Check the `outputs/` folder.

## Training

`COMING SOON`

1. Run `python train.py`.
2. Find your new model in `models/`.
2. Run `python inference.py` but load your own model.
3. Check the `outputs/` folder.

## Reminder 

Remember, the [project website](https://relh.github.io/FAE-HMI-SI/) has detailed explanations and demos:

<p align="center">
<img src="./website/assets/website_screengrab.png" alt="reminder project website screengrab" width="600"/>
</p>

Credit to [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) for the great pytorch unet implementation!

Credit to [js-image-zoom](https://github.com/malaman/js-image-zoom) for the great image zoom tool on the project website.
