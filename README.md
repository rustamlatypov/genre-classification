# Machine learning classifier for music genres
Project for the Aalto university course CS-E3210 - Machine Learning: Basic Principles (fall 2018).

## Abstract
This data analysis project focuses on a multiclass music genre classification problem. Pre-extracted summary characteristics of the data are further preprocessed using a median absolute deviation based feature selection method and supervised principal component analysis. The prediction power of a support vector machine is optimized using micro-averaged f1-score and multiclass logarithmic loss as evaluation metrics. Succesful predictions are made for the labels of the majority classes, while compromising minority class prediction accuracy. This is deemed acceptable as the data is significantly skewed.

## Kaggle

This classifier placed in the top 8% in two Kaggle competions. 

- Accuracy: 65.2%, Place: 30/406, [evaluation metric: accuracy](https://www.kaggle.com/c/mlbp-data-analysis-challenge-accuracy-2018/leaderboard)
- Log-loss: 0.1698, Place: 26/371, [evaluation metric: log-loss](https://www.kaggle.com/c/mlbp-data-analysis-challenge-log-loss-2018/leaderboard)


## Instructions for viewing
One must be able to compile a Jupyter Notebook for direct viewing. If code is extracted as a stand-alone file, a separate Python compiler must be used. 

Running cells is not nessesery for viewing, but if one wishes to run the cells, the following files must be included in the root directory: 

1. get.py (auxiliary functions)
2. sPCA.py (sPCA implementation)
3. test_data.csv
4. train_data.csv
5. train_labels.csv


## Prerequisites
Built on Jupyter Notebook 7.2.0 and Python 3.7

## Authors

- [Rustam Latypov](mailto:rustam.latypov@aalto.fi)
- [Kalle Alaluusua](mailto:kalle.alaluusua@aalto.fi)
