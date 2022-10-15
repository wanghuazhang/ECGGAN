# ECGGAN

## Overview
This is the code implementation for ECGGAN: An Electrocardiogram Anomaly Detection Framework Based on Reconstruction of Periodic Metadata. 


## Requirements
- python 3.8.8
- pytorch 1.8.1
- heartpy 1.2.7
- cudatoolkit 10.2.89
- numpy 1.19.2 
- scikit-learn 0.24.1

## DataSet
- CPSC: Feifei Liu, Chengyu Liu, Lina Zhao, Xiangyu Zhang, Xiaoling Wu, Xiaoyan Xu, Yulin Liu, Caiyun Ma, Shoushui Wei, Zhiqiang He, et al. 2018. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. Journal of Medical Imaging and Health Informatics 8, 7 (2018), 1368-1373. http://2018.icbeb.org/Challenge.html

- AIWIN: AIWIN 2021. AIWIN (Autumn) - ECG Intelligent Diagnosis Competition by SID Medical and Fudan University Zhongshan Hospital. http://ailab.aiwin.org.cn/competitions/64#learn_the_details

## Usage

### Data Preparation
For ECGGAN full experiemnt on CPSC, AIWIN, and the mixed-set (need to download full dataset first)

`python preProcess.py`

```
put CPSC and AIWIN files under dataset

dataset
|AIWIN
| | raw_ECGdata            # The raw ECG files
| | normal_artificial.csv  # Annotate the noise leads in each normal ECG
| | reference.csv          # raw ECGs label
|CPSC
| | raw_ECGdata            # The raw ECG files
| | normal_artificial.csv  # Annotate the noise leads in each normal ECG
| | reference.csv          # raw ECGs label

```

## Reference
