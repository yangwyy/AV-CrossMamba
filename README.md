## Project Structure

`/data/voxceleb2-800`: Scripts to preprocess the voxceleb2 datasets.

`/pretrain_networks`: The visual front-end network

`/src`: The training scripts

## Pre-trained Weights
Download the pre-trained weights for the Visual Frontend and place it in the ./pretrain_networks folder using the following command:

	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k0Zk90ASft89-xAEUbu5CmZWih_u_lRN' -O visual_frontend.pt


## References
1. The pre-trained weights of the Visual Frontend have been obtained from [Afouras T. and Chung J, Deep Audio-Visual Speech Recognition](https://github.com/lordmartian/deep_avsr) GitHub repository.

2. The model is adapted from [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) GitHub repository.

## Requirements

	Python >= 3.8
	PyTorch >= 2.5.0
	CUDA >= 12.4
	Other dependencies in requirements.txt

## Installation

	git clone https://github.com/yangwyy/AV-CrossMamba.git
	cd AV-CrossMamba
	pip install torch torchvision
	pip install -r requirements.txt
