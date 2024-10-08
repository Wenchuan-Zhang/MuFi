# MuFi: Multimodal Full information Model

This repository contains the PyTorch implementation of **MuFi**, a model designed for multi-feature integration, specifically optimized for predicting clinical outcomes in breast cancer. MuFi combines radiomics, pathomics, and clinical features to enhance the model's predictive capabilities. This repository also includes necessary scripts, requirements, and instructions for reproducing the results from our study.

## Features
- **Multi-modal data integration**: The model leverages different feature sets, including radiomics, pathomics, and clinical data.
- **Customizable architecture**: The model architecture is modular and can be easily adapted or extended with additional feature sets.
- **Detailed evaluation**: Supports a variety of metrics such as AUC, accuracy, sensitivity, specificity, PPV, and NPV, providing a comprehensive performance assessment.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Wenchuan-Zhang/MuFi.git
   cd MuFi
