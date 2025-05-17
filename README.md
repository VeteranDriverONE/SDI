# SDI
A method for predicting whether the co-assembly of acoustic sensitizers and chemotherapy drugs is nanoparticles. 

This is the source code of the paper titled "***Construction of Sonosensitizer-drug Co-assembly Based on Deep Learning Method***". The main.py serves as the entry point for training, the test_new.py is the entry point for testing, and the test_vis_att.py is used to visualize the attention of the model on the nodes.

# Data Format

In our model, two types of files need to be set as inputs:
* Molecular dictionary: It is used to store the properties of each molecular.
* Dataset: It is used to list the paired sonosensitizers and chemotherapy drugs, as well as the labels of these pairs.

Dictionary of Sonosensitizers or Chemotherapy:
It is a CSV file, as shown in *dataset/selected_drugs_smiles-sample.csv*. Each row represents a molecule, and there are three columns in total, namely NAME (the name of the drug molecule), SMILES (the SMILES code of the drug molecule), and SIZE (the measured average particle size of the molecular).

Training dataset, validation dataset, or test dataset:
It is a CSV file with four columns: sonosensitizer, drug, size, and classes, as shown in *dataset/traindata31-sample.csv*. They represent the name of the sonosensitizer, the name of the chemotherapy drug, the size after co - assembly, and the class respectively. The class has two categories, indicating whether it is a nanoparticle or not. Note: The names of sonosensitizers and chemotherapy drugs should be consistent with those in the molecular dictionary.


# Citation Information
**Title**: Construction of Sonosensitizer‐Drug Co‐Assembly Based on Deep Learning Method (https://onlinelibrary.wiley.com/doi/10.1002/smll.202502328)

**Abstract**: 
Drug co-assemblies have attracted extensive attention due to their advantages of easy preparation, adjustable performance and drug component co-delivery. However, the lack of a clear and reasonable co-assembly strategy has hindered the wide application and promotion of drug-co assembly. This paper introduces a deep learning-based sonosensitizer-drug interaction (SDI) model to predict the particle size of the drug mixture. To analyze the factors influencing the particle size after mixing, the graph neural network was employed to capture the atomic, bond, and structural features of the molecules. A multi-scale cross-attention mechanism is designed to integrate the feature representations of different scale substructures of the two drugs, which not only improves prediction accuracy but also allows for the analysis of the impact of molecular structures on the predictions. Finally, ablation experiments are conducted to assess the influence of various molecular properties on prediction accuracy. The proposed method is compared against both machine learning and deep learning approaches. The experimental results demonstrate that the SDI model achieves a precision of 90.00%, a recall rate of 96.00%, and an F1-score of 91.67%, surpassing the performance of the comparative machine learning and deep learning methods. Furthermore, the SDI predicts the co-assembly of the chemotherapy drug methotrexate (MET) and the sonosensitizer emodin (EMO) to form the nanomedicine NanoME. This prediction is further validated through experiments, demonstrating that NanoME can be used for fluorescence imaging of liver cancer and sonodynamic/chemotherapy anticancer therapy.

**Keywords**: drug co-assembly; particle size prediction; graph neural network; deep learning; theranostics

![image](TOC.png)

Cite this paper:
```
@article{https://doi.org/10.1002/smll.202502328,
author = {Wang, Kanqi and Yang, Liuyin and Lu, Xiaowei and Cheng, Mingtao and Gui, Xiran and Chen, Qingmin and Wang, Yilin and Zhao, Yang and Li, Dong and Liu, Gang},
title = {Construction of Sonosensitizer-Drug Co-Assembly Based on Deep Learning Method},
journal = {Small},
volume = {n/a},
number = {n/a},
pages = {2502328},
```