# Machine learning classifier for music genres

## Abstract
This data analysis project focuses on a multiclass musical genre classification problem. Pre-extracted summary characteristics of the data are further preprocessed using a median absolute deviation based feature selection method and supervised principal component analysis. The prediction power of a support vector machine is optimized using micro-averaged f1-score and multiclass logarithmic loss as evaluation metrics. We succesfully predict the labels of the majority classes, while compromising minority class prediction accuracy. This is deemed acceptable as the data are significantly skewed.

## Instructions for viewing
One must be able to compile IPython Notebooks for effortles reading. If code is extracted to a stand-alone file, a separate compiler must be used. 

Running cells is not nessesery for viewing, but if one wishes to run the cells, the following files must be included in the root directory: 

1. get.py (auxiliary functions)
2. sPCA.py (own sPCA implementation)
3. test_data.csv
4. train_data.csv
5. train_labels.csv


## Prerequisites
- Built on IPython Notebook 7.2.0 and Python 3.7

## Authors

- [Rustam Latypov](mailto:rustam.latypov@aalto.fi)
- [Kalle Alaluusua](mailto:kalle.alaluusua@aalto.fi)
