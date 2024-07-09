# ASG
[![License:Physionet](https://img.shields.io/badge/License-Physionet-red.svg)](https://physionet.org/)
[![Homepage](https://img.shields.io/badge/Homepage-Visit-blue)](https://asgmvlp.github.io/)



The official code for **Anatomical Structure-Guided Medical Vision-Language Pre-training**.

- **[2024.06]** ASGMVLP is accepted by MICCAI 2024 🎉!

## Installation
```
# Set up the environment
conda create --name asgmvlp python=3.8.5

# Activate the environment
conda activate asgmvlp

# Install required packages
pip install -r requirements.txt
```
## Dataset
**Pre-training Dataset**
> We pre-train our ASG framework on the JPG version of MIMIC-CXR 2.0.0 dataset. For each image, we resize the larger size to 256 and pad zeros on the smaller side, which results in the image size of 256 × 256. During training, we randomly crop a 224 × 224 image.
- [**MIMIC-CXR-JPG.**](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) 217k image-text pairs.

**Finetune Dataset**
> We follow the data split and metrics (AUC/ACC) from MGCA. Since MGCA does not conduct experiments on the NIH X-ray, we follow KAD's split.
- [**CheXpert.**]( https://stanfordmlgroup.github.io/competitions/chexpert/) We use the original validation set as test data and randomly select 5, 000 radiographs from training data for validation. 
- [**RSNA.**](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) We manually split the dataset into training, validation, and test set with 70%/15%/15% ratio.
- [**COVIDx.**](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) We use the original validation set as test data and split 10% of original training set for validation.
- [**NIH X-ray.**]( https://nihcc.app.box.com/v/ChestXray-NIHCC) We use the original validation set as test data and split 10% of original training set for validation.
- [**SIIM.**](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data)  We manually split the dataset into training, validation, and test set with 70%/15%/15% ratio.

## Pre-training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --learning_rate 4e-5 --batch_size 72 --data_dir /path/to/mimic-cxr --output_dir /path/to/save/logs
```
## Results
**Linear Probe Classification**
<table >
    <caption>Comparison with other SOTA methods on the classification task.</caption>
    <thead>
        <tr>
            <th rowspan="2">Method</th>
            <th colspan="3">NIH X-ray (AUC)</th>
            <th colspan="3">CheXpert (AUC)</th>
            <th colspan="3">RSNA (AUC)</th>
            <th colspan="3">COVIDx (ACC)</th>
        </tr>
        <tr>
            <th>1%</th>
            <th>10%</th>
            <th>100%</th>
            <th>1%</th>
            <th>10%</th>
            <th>100%</th>
            <th>1%</th>
            <th>10%</th>
            <th>100%</th>
            <th>1%</th>
            <th>10%</th>
            <th>100%</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Random Init</td>
            <td>52.1</td>
            <td>54.6</td>
            <td>55.3</td>
            <td>56.1</td>
            <td>62.6</td>
            <td>65.7</td>
            <td>58.9</td>
            <td>69.4</td>
            <td>74.1</td>
            <td>50.5</td>
            <td>60.3</td>
            <td>70.0</td>
        </tr>
        <tr>
            <td>ImageNet Init</td>
            <td>67.0</td>
            <td>67.5</td>
            <td>71.6</td>
            <td>74.4</td>
            <td>79.7</td>
            <td>81.4</td>
            <td>74.9</td>
            <td>74.5</td>
            <td>76.3</td>
            <td>64.8</td>
            <td>78.8</td>
            <td>86.3</td>
        </tr>
        <tr>
            <td><em>CNN-based</em></td>
            <td colspan="12"></td>
        </tr>
        <tr>
            <td>ConVIRT </a ></td>
            <td>64.9</td>
            <td>77.1</td>
            <td>80.8</td>
            <td>85.9</td>
            <td>86.8</td>
            <td>87.3</td>
            <td>77.4</td>
            <td>80.1</td>
            <td>81.3</td>
            <td>72.5</td>
            <td>82.5</td>
            <td>92.0</td>
        </tr>
        <tr>
            <td>GLoRIA </td>
            <td>59.7</td>
            <td>74.3</td>
            <td>80.0</td>
            <td>87.1</td>
            <td>88.7</td>
            <td>88.0</td>
            <td>87.0</td>
            <td>89.4</td>
            <td><strong>90.2</strong></td>
            <td>66.5</td>
            <td>80.5</td>
            <td>88.0</td>
        </tr>
        <tr>
            <td>MedKLIP </td>
            <td>60.9</td>
            <td>74.8</td>
            <td>80.1</td>
            <td>82.3</td>
            <td>85.4</td>
            <td>87.3</td>
            <td>83.3</td>
            <td>86.6</td>
            <td>88.1</td>
            <td>74.5</td>
            <td>83.5</td>
            <td>91.3</td>
        </tr>
        <tr>
            <td>MedCLIP</td>
            <td>76.5</td>
            <td>80.5</td>
            <td>82.1</td>
            <td>87.1</td>
            <td>87.6</td>
            <td>88.1</td>
            <td>87.0</td>
            <td>88.6</td>
            <td>89.2</td>
            <td>73.5</td>
            <td>82.3</td>
            <td>91.3</td>
        </tr>
        <tr>
            <td>KAD </td>
            <td>78.7</td>
            <td>80.7</td>
            <td>82.5</td>
            <td>87.2</td>
            <td>88.6</td>
            <td>88.7</td>
            <td>86.7</td>
            <td>88.7</td>
            <td>89.9</td>
            <td>73.5</td>
            <td>83.0</td>
            <td>90.5</td>
        </tr>
        <tr>
            <td>MGCA </td>
            <td>77.7</td>
            <td>80.8</td>
            <td>82.6</td>
            <td>87.6</td>
            <td>88.0</td>
            <td>88.2</td>
            <td>87.6</td>
            <td>88.6</td>
            <td>89.8</td>
            <td>72.0</td>
            <td>83.5</td>
            <td>90.5</td>
        </tr>
        <tr>
            <td>Ours</td>
            <td>77.0</td>
            <td>81.0</td>
            <td>82.9</td>
            <td>87.7</td>
            <td>88.2</td>
            <td>88.7</td>
            <td>87.2</td>
            <td><u>88.8</u></td>
            <td>89.7</td>
            <td>77.3</td>
            <td>84.8</td>
            <td><strong>93.3</strong></td>
        </tr>
        <tr>
            <td><em>ViT-based</em></td>
            <td colspan="12"></td>
        </tr>
        <tr>
            <td>MRM</td>
            <td>78.0</td>
            <td><u>82.1</u></td>
            <td>83.2</td>
            <td><u>88.5</u></td>
            <td>88.5</td>
            <td>88.7</td>
            <td>87.2</td>
            <td>88.7</td>
            <td>89.7</td>
            <td><u>79.0</u></td>
            <td><u>85.5</u></td>
            <td><u>92.5</u></td>
        </tr>
        <tr>
            <td>MGCA </td>
            <td><u>78.9</u></td>
            <td><u>82.1</u></td>
            <td><u>83.5</u></td>
            <td><strong>88.8</strong></td>
            <td><strong>89.1</strong></td>
            <td><strong>89.7</strong></td>
            <td><strong>88.6</strong></td>
            <td><strong>89.5</strong></td>
            <td><u>90.0</u></td>
            <td>74.8</td>
            <td>84.8</td>
            <td>92.3</td>
        </tr>
        <tr>
            <td>Ours</td>
            <td><strong>79.5</strong></td>
            <td><strong>82.2</strong></td>
            <td><strong>83.6</strong></td>
            <td>87.9</td>
            <td><u>89.0</u></td>
            <td><u>89.0</u></td>
            <td><u>88.4</u></td>
            <td><strong>89.5</strong></td>
            <td><strong>90.2</strong></td>
            <td><strong>81.3</strong></td>
            <td><strong>87.0</strong></td>
            <td><strong>93.3</strong></td>
        </tr>
    </tbody>
</table>

**Zero-Shot Classification**
> For zero-shot, we use the same data split as in the linear probe.
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="3">RSNA</th>
      <th colspan="3">NIH X-ray</th>
    </tr>
    <tr>
      <th>AUC</th>
      <th>F1</th>
      <th>ACC</th>
      <th>AUC</th>
      <th>F1</th>
      <th>ACC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BioViL</td>
      <td>83.8</td>
      <td>58.1</td>
      <td>77.8</td>
      <td>73.8</td>
      <td>25.2</td>
      <td>85.9</td>
    </tr>
    <tr>
      <td>MedKLIP</td>
      <td>84.5</td>
      <td>61.1</td>
      <td>74.2</td>
      <td>75.6</td>
      <td>26.0</td>
      <td>87.8</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td><b>86.2</b></td>
      <td><b>62.8</b></td>
      <td><b>79.4</b></td>
      <td><b>77.0</b></td>
      <td><b>27.5</b></td>
      <td><b>90.0</b></td>
    </tr>
  </tbody>
  </table>

**Segmentation**
<table>
    <tr>
        <th rowspan="2">Method</th>
        <th colspan="3">SIIM (Dice)</th>
        <th colspan="3">RSNA (Dice)</th>
    </tr>
    <tr>
        <th>1%</th>
        <th>10%</th>
        <th>100%</th>
        <th>1%</th>
        <th>10%</th>
        <th>100%</th>
    </tr>
    <tr>
        <td>Random Init</td>
        <td>9.00</td>
        <td>28.6</td>
        <td>54.3</td>
        <td>6.90</td>
        <td>10.6</td>
        <td>18.5</td>
    </tr>
    <tr>
        <td>ImageNet Init</td>
        <td>10.2</td>
        <td>35.5</td>
        <td>63.5</td>
        <td>34.8</td>
        <td>39.9</td>
        <td>64.0</td>
    </tr>
    <tr>
        <td colspan="7"><i>CNN-based</i></td>
    </tr>
    <tr>
        <td>ConVIRT</td>
        <td>25.0</td>
        <td>43.2</td>
        <td>59.9</td>
        <td>55.0</td>
        <td>67.4</td>
        <td>67.5</td>
    </tr>
    <tr>
        <td>GLoRIA</td>
        <td>37.4</td>
        <td>57.1</td>
        <td>64.2</td>
        <td>60.3</td>
        <td>68.7</td>
        <td>68.3</td>
    </tr>
    <tr>
        <td>MedKLIP</td>
        <td>55.1</td>
        <td>62.0</td>
        <td>66.8</td>
        <td>64.7</td>
        <td>68.9</td>
        <td>70.3</td>
    </tr>
    <tr>
        <td>MedCLIP</td>
        <td>51.2</td>
        <td>62.6</td>
        <td>67.6</td>
        <td>65.7</td>
        <td>68.6</td>
        <td>69.6</td>
    </tr>
    <tr>
        <td>KAD</td>
        <td>58.4</td>
        <td>68.2</td>
        <td>69.9</td>
        <td>67.9</td>
        <td>68.5</td>
        <td>70.3</td>
    </tr>
    <tr>
        <td>MGCA</td>
        <td>49.7</td>
        <td>59.3</td>
        <td>64.2</td>
        <td>63.0</td>
        <td>68.3</td>
        <td>69.8</td>
    </tr>
    <tr style="background-color: #D3D3D3;">
        <td>Ours</td>
        <td>60.7</td>
        <td>66.7</td>
        <td><u>73.6</u></td>
        <td>68.4</td>
        <td>69.9</td>
        <td><u>72.6</u></td>
    </tr>
    <tr>
        <td colspan="7"><i>ViT-based</i></td>
    </tr>
    <tr>
        <td>MRM</td>
        <td><u>68.3</u></td>
        <td><u>69.5</u></td>
        <td>72.2</td>
        <td><u>69.5</u></td>
        <td>69.2</td>
        <td>70.6</td>
    </tr>
    <tr>
        <td>MGCA</td>
        <td>60.1</td>
        <td>65.4</td>
        <td>69.6</td>
        <td>69.3</td>
        <td><u>70.0</u></td>
        <td>72.3</td>
    </tr>
    <tr style="background-color: #D3D3D3;">
        <td>Ours</td>
        <td><b>71.9</b></td>
        <td><b>74.7</b></td>
        <td><b>75.6</b></td>
        <td><b>71.7</b></td>
        <td><b>72.3</b></td>
        <td><b>72.8</b></td>
    </tr>
</table>

**Ablation Study**
<table>
<thead>
  <tr>
    <th colspan="4">Learning Objective</th>
    <th colspan="3">NIH X-ray (AUC)</th>
    <th colspan="3">COVIDx (ACC)</th>
    <th colspan="3">RSNA (Dice)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>IRA</b></td>
    <td><b>ARSA</b></td>
    <td><b>IRL</b></td>
    <td><b>ERL</b></td>
    <td><b>1%</b></td>
    <td><b>10%</b></td>
    <td><b>100%</b></td>
    <td><b>1%</b></td>
    <td><b>10%</b></td>
    <td><b>100%</b></td>
    <td><b>1%</b></td>
    <td><b>10%</b></td>
    <td><b>100%</b></td>
  </tr>
  <tr>
    <td>√</td>
    <td></td>
    <td></td>
    <td></td>
    <td>78.2</td>
    <td>81.7</td>
    <td>82.6</td>
    <td>75.3</td>
    <td>85.8</td>
    <td>91.0</td>
    <td>65.1</td>
    <td>67.7</td>
    <td>68.3</td>
  </tr>
  <tr>
    <td>√</td>
    <td>√<sup>&dagger;</sup></td>
    <td></td>
    <td></td>
    <td>79.1</td>
    <td>81.8</td>
    <td>83.1</td>
    <td>77.5</td>
    <td>86.0</td>
    <td>92.3</td>
    <td>70.6</td>
    <td>71.2</td>
    <td>71.9</td>
  </tr>
  <tr>
    <td>√</td>
    <td>√<sup>#</sup></td>
    <td></td>
    <td></td>
    <td>78.9</td>
    <td>81.5</td>
    <td>83.4</td>
    <td>76.3</td>
    <td>86.3</td>
    <td>92.0</td>
    <td>69.0</td>
    <td>69.4</td>
    <td>69.7</td>
  </tr>
  <tr>
    <td>√</td>
    <td></td>
    <td>√</td>
    <td></td>
    <td>78.7</td>
    <td>81.8</td>
    <td>82.9</td>
    <td>78.3</td>
    <td>86.0</td>
    <td>91.0</td>
    <td>66.2</td>
    <td>68.6</td>
    <td>68.8</td>
  </tr>
  <tr>
    <td>√</td>
    <td></td>
    <td>√</td>
    <td>√</td>
    <td>78.8</td>
    <td>81.7</td>
    <td>83.4</td>
    <td>79.3</td>
    <td>86.5</td>
    <td>92.8</td>
    <td>67.4</td>
    <td>68.6</td>
    <td>69.7</td>
  </tr>
  <tr>
    <td>√</td>
    <td>√<sup>&dagger;</sup></td>
    <td>√</td>
    <td>√</td>
    <td><b>79.5</b></td>
    <td><b>82.2</b></td>
    <td><b>83.6</b></td>
    <td><b>81.3</b></td>
    <td><b>87.0</b></td>
    <td><b>93.3</b></td>
    <td><b>71.7</b></td>
    <td><b>72.3</b></td>
    <td><b>72.8</b></td>
  </tr>
</tbody>
</table>

## Schedule
- [ ] Release the alignment rules and re-labeled datasets. 
- [ ] More details …
## Acknowledgements
This project is built upon [MGCA](https://github.com/HKU-MedAI/MGCA). Thanks to their great contribution!



