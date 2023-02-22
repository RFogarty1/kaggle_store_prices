# Kaggle Store Sales Prediction

These are some notebooks associated with the Kaggle ["Store Sales - Time Series Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

# Structure

**eda** - "Exploratory Data Analysis". Some plots showing average sales, and oil prices (init_explore.ipynb). Also a notebook showing mutual information between features and targets.

**model** - Notebook showing effects of changing features on a random forest model.

**submission** - Contains a single notebook which writes a file which can be submitted to Kaggle


# Setup

Notebooks can easily be browsed online or in the jupyter notebook environment.

Assuming a linux OS with pyenv installed, the bash commands to setup a python virtual environment with the required dependencies can be found in install\_deps.sh.

Furthermore, the required competition data needs to be moved into the raw\_data folder. This data can be found [here](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). For example the training data should be located at "raw\_data/train.csv" (relative to the base directory).



