![test](https://github.com/andriygav/MixtureLib/workflows/test/badge.svg?branch=master)
![.github/workflows/docs.yml](https://github.com/andriygav/MixtureLib/workflows/.github/workflows/docs.yml/badge.svg?branch=master)

# MixtureLib

## Requirements
1. Python 3.6.2
2. pip 19.2.3

## Installation
```
git clone https://github.com/andriygav/MixtureLib.git
cd MixtureLib
python3.6 -m pip install ./src/
```

## Uninstallation
```
python3.6 -m pip uninstall MixtureLib
```

## Version
0.0.1

## Example
Example file is available [here](https://github.com/andriygav/MixtureLib/blob/master/examples/example.ipynb). Before launch this file, please install MixtureLib lib.
### Mixture Of Experts
Mixture of Expert allow us to get model which are correspond to current point.
![Mixture of Experts Leaning image](https://github.com/andriygav/MixtureLib/raw/master/examples/pictures/pi_predicftion_experts.png)

### Mixture Of Models
Mixture of Models don't allow us to find best model for each point.
![Mixture of Models Leaning image](https://github.com/andriygav/MixtureLib/raw/master/examples/pictures/pi_predicftion_models.png)

### Comparison
Convergence of local models for different SuperModels are illustrated here. Both models converge to the real point.

Mixture of Experts             |  Mixture of Models
:-------------------------:|:-------------------------:
![Mixture of Experts Leaning parameters](https://raw.githubusercontent.com/andriygav/MixtureLib/master/examples/pictures/parameters_experts.png)  |  ![Mixture of Models Leaning parameters](https://raw.githubusercontent.com/andriygav/MixtureLib/master/examples/pictures/parameters_models.png)


