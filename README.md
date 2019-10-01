# Using supervised PCA with SVM for music genre classification

Project for the Aalto University course CS-E3210 - Machine Learning: Basic Principles.

Developed during October - November, 2018.

## Description

This data analysis project focuses on a multiclass music genre classification problem. Pre-extracted summary characteristics of the data are further preprocessed using a median absolute deviation based feature selection method and supervised principal component analysis. A support vector machine is trained on the data and optimized using a micro-averaged f1-score and multiclass logarithmic loss as evaluation metrics. The majority classes are predicted successfully, while compromising the prediction accuracy of the minority classes. This is deemed acceptable as the data is significantly skewed.


## Running

Built for Jupyter Notebook 7.2.0 with Python 3.7.2.

One must be able to compile a Jupyter Notebook for direct viewing. If code is extracted as a stand-alone file, a separate Python interpreter is needed.

Running cells is not necessary for viewing, but if one wishes to run the cells, the following files must be included in the root directory: 

- get.py (auxiliary functions)
- sPCA.py (sPCA implementation)
- test_data.csv
- train_data.csv
- train_labels.csv

## Kaggle

This classifier placed in the top 8% in two Kaggle competions. 

- Accuracy: 65.2%, Place: 30/406, [evaluation metric: accuracy](https://www.kaggle.com/c/mlbp-data-analysis-challenge-accuracy-2018/leaderboard)
- Log-loss: 0.1698, Place: 26/371, [evaluation metric: log-loss](https://www.kaggle.com/c/mlbp-data-analysis-challenge-log-loss-2018/leaderboard)


## Authors

- [Rustam Latypov](mailto:rustam.latypov@aalto.fi)
- Kalle Alaluusua
