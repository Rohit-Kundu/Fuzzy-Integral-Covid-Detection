# Fuzzy-Integral-Ensemble
This is the official implementation of the paper titled "Fuzzy Integral-based CNN Ensemble for COVID-19 Detection from Lung CT Images" accepted for publication in _Computers in Biology and Medicine_, Elsevier.

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
