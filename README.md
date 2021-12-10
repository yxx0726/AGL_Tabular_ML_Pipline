# AutoGluon-Tabular Downstream Benchmarking Pipline
Pipeline for benchmarking AutoGluon's downsteam model performance with different feature type inference strategies.  
Downstream models: RandomForest, XGBoost and NN-FastAI  
Feature type inference strategies: Truth, AGL and AGL+SH  

## Links
Forked SortingHat: https://github.com/fabulousdj/SortingHatLib  
Forked AutoGluon: https://github.com/fabulousdj/autogluon  

## How to install?
This project requires installation of SortingHat and AutoGluon using the links provided above. These are forked from their original repos and modified to accomondate the study of improving downstream model performance of AutoGluon with SortingHat.

To start off, you may need a fresh Python venv with Python 3.8. Other Python versions may cause incompatibility with AutoGluon.
Install the forked SortingHat by following instructions in the README or shown below:
```
git clone https://github.com/fabulousdj/SortingHatLib.git
pip install SortingHatLib/
```
Install NLTK package via:
```
pip install nltk
```
Install the forked AutoGluon by following instructions below:
```
python3 -m pip install -U pip
python3 -m pip install -U setuptools wheel
python3 -m pip install -U "mxnet<2.0.0"
git clone https://github.com/fabulousdj/autogluon
cd autogluon && ./full_install.sh
```
(Note that AutoGluon currently does not fully support Apple M1 chips. It would be ideal to install and run on a local or remote Linux machine or Apple computers with Intel chips to avoid installation errors.)

## How to run?
Once you finished the installation steps above, you can launch the TabularPipelineV2.py script and execute with your target settings by modifying the code in the main function. An example run has been provided to compare model performance on regression tasks with CA datasets.

## Having issues?
Contact us at:  
Jin Dai: jidai@ucsd.edu  
Xin Yu: xiy264@ucsd.edu
