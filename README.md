# Fuzzy-Integral-Ensemble
This is the official implementation of the paper titled "Fuzzy Integral based CNN Ensemble for COVID-19 Detection from Lung CT Images".

Abstract: The COVID-19 pandemic has collapsed the public healthcare systems, along with severely damaging the economy of the world. The SARS-CoV-2 virus also known as the coronavirus, originated in Wuhan, China and led to community spread, causing the death of more than a million people worldwide. The primary reason for the uncontrolled spread of the virus is the lack of provision for population-wise screening. The apparatus for RT-PCR based COVID-19 detection is scarce and the testing process takes 6-9 hours. The test is also not satisfactorily sensitive (71\% sensitive only). Hence, Computer-Aided Detection techniques based on Deep Learning methods can be used in such a scenario using other modalities like chest CT-scan images for more accurate and sensitive screening. In this paper, we propose a method that uses a Sugeno fuzzy integral ensemble of four pre-trained Deep Learning models, namely, VGG-11, GoogLeNet, SqueezeNet v1.1 and Wide ResNet-50-2, for classification of chest CT-scan images into COVID and Non-COVID categories. The proposed framework has been tested on a publicly available dataset for evaluation and it achieves 98.93\% accuracy and 98.93\% sensitivity on the same. The model outperforms state-of-the-art methods on the same dataset and proves to be a reliable COVID-19 detector.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the COVID data

Download the dataset from [Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) and split it into train and validation sets in 80-20 ratio.

Required Directory Structure:
```

.
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- main.py
+-- probability_extraction.py
+-- ensemble.py
+-- sugeno_integral.py

```

Run: `python main.py --data_directory "D:/data" --epochs 100`
