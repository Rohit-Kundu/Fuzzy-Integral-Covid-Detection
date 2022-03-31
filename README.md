# Fuzzy-Integral-Ensemble
This is the official implementation of the paper titled "[Fuzzy Integral-based CNN Ensemble for COVID-19 Detection from Lung CT Images](https://doi.org/10.1016/j.compbiomed.2021.104895)" published in _Computers in Biology and Medicine_, Elsevier.

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

# Citation
If you found this repository helpful, please consider citing our paper:
```
@article{kundu2021covid,
  title={COVID-19 detection from lung CT-Scans using a fuzzy integral-based CNN ensemble},
  author={Kundu, Rohit and Singh, Pawan Kumar and Mirjalili, Seyedali and Sarkar, Ram},
  journal={Computers in Biology and Medicine},
  volume={138},
  pages={104895},
  year={2021},
  publisher={Elsevier}
}
```
