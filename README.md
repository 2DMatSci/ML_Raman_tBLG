# Machine Learning Raman tBLG

This repository holds the example code and reduced dataset to determine the twist angle of twisted bilayer graphene (tBLG) from its Raman spectrum. The code is a functional version of that used in the published paper (_ACS Applied Nano Materials_ 5, 1356-1366, **2022** doi:10.1021/acsanm.1c03928).

## Usage

Start ml_raman_tblg.py with the arguments "train" or "predict":

* For training the file containing the training dataset must be specified (train_dataset.csv) along with the desired ML algorithm and scaler (ml_raman_tblg.py train -h for help on usage). The trained model will be saved in a file.
* For predicting, the files with the saved trained model and the dataset to predict must be provided (ml_raman_tblg.py predict -h for help on usage).

All the output files will be saved in the "results" folder.

For testing purposes, the complete training dataset (train_dataset.csv) is included, along with the datasets from figures 4d and 4g of _ACS Applied Nano Materials_ 5, 1356-1366, **2022** doi:10.1021/acsanm.1c03928.

## License

All code found and data in this repository is licensed under GPL v3

```
Copyright 2022 Pablo Solís-Fernández

This file is part of Machine Learning Raman tBLG

Machine Learning Raman tBLG is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

Machine Learning Raman tBLG is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with Machine Learning Raman tBLG. If not, see <http://www.gnu.org/licenses/>.
```

## Requirements

* Scikit-learn 1.0
* Pandas 1.2.4

## Citations

The detailed results are provided in the following paper:

* "_Machine Learning Determination of the Twist Angle of Bilayer Graphene: Implications for Twisted van der Waals Heterostructures_", P. Solís-Fernández and H. Ago, _ACS Applied Nano Materials_ 5, 1356-1366, **2022** doi:10.1021/acsanm.1c03928.

If you find this useful, please consider citing the paper in your research.
