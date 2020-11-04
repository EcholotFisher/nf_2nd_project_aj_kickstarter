# nf_2nd_project_aj_kickstarter
This project uses a dataset from Kickstarter to explore prediction models.
We have created a Conda environment for this purpose.

### Load the package using the below command
    conda env create --file=environment.yml
    conda activate nf_2nd_project
This will load all the packages listed at the bottom of this README

### The Relevant files in this repo are:
EDA_brushup.ipynb : contains our EDA
modelling_notebook.ipynb : contains our exploratory modelling
train.py : script for training a model
predict.py : script for predicting values using a test data set
feature_engineering.py : library of functions used in train and predicitions

### To train the model run the following (in our case data_folder is 'kickstarter/data'):
    python train.py <data_folder>

### To make predictions using a dataset (one is provided, replace filename as necessary) use the following command:
    python predict.py models/DecisionTreeClassifier_model.sav test_data/X_test.csv test_data/y_test.csv

# Conda environment
### packages in environment.yml:

name: nf_2nd_project
channels:
  - defaults
dependencies:
  - python=3.8.5
  - pandas
  - jupyterlab
  - nb_conda_kernels
  - scikit-learn
  - matplotlib
  - seaborn
prefix: /Users/andrewemmett/opt/anaconda3/envs/nf_2nd_project
