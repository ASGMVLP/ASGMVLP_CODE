# ASG
The official code for **Anatomical Structure-Guided Medical Vision-Language Pre-training**.
## Installation
```
# Set up the environment
conda create --name asgmvlp python=3.8.5

# Activate the environment
conda activate mimiccxrvqa

# Install required packages
pip install -r requirements.txt
```
## Dataset
**Pre-training Dataset**
> We pre-train our ASG framework on the JPG version of MIMIC-CXR 2.0.0 dataset. For each image, we resize the larger size to 256 and pad zeros on the smaller side, which results in the image size of 256 × 256. During training, we randomly crop an 224 × 224 image.
- [**MIMIC-CXR-JPG.**](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) 217k image-text pairs.

**Finetune Dataset**
> We follow the data split from MGCA and GLoRIA. Since they does not conduct experiments on the NIH ChestXray14 Dataset, we follow KAD's split.
- [**CheXpert.**]( https://stanfordmlgroup.github.io/competitions/chexpert/) We use the original validation set as test data and randomly select 5, 000 radiographs from training data for validation. 

- [**RSNA.**](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) We manually split the dataset into training, validation, and test set with 70%/15%/15% ratio.
 
- [**COVIDx.**](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) We use the original validation set as test data and split 10% of original training set for validation.

- [**NIH ChestXray14.**]( https://nihcc.app.box.com/v/ChestXray-NIHCC) We use the original validation set as test data and split 10% of original training set for validation.

- [**SIIM.**](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data)  We manually split the dataset into training, validation, and test set with 70%/15%/15% ratio.

## Pre-training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --learning_rate 4e-5 --batch_size 72 --data_dir /path/to/mimic-cxr --output_dir /path/to/save/logs
```
## Results
**Linear Probe Classification**

**Zero-Shot Classificatin**

**Segmentation**

## Schedule
- [ ] Realse the alignment rules and re-labeled datasets. 
- [ ] More details …
## Acknowledgements
This project is built upon [MGCA](https://github.com/HKU-MedAI/MGCA). Thanks to their great contribution!



