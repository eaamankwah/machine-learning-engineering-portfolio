# Bertelsmann-Arvato-Capstone-Project

## Overview
The capstone project is part of the final project for the completion of Udacity's Machine Learning Engineer Nanodegree program.

Arvato Financial Solutions which is an arm of Bertelsmann company made available the private data and specific instructions to formulate the business problem while Udacity provided the workplace environment and a starting Jupyter Notebook for the capstone project. The challenge was how to increase the efficiency of the customer acquisition process of a mail-order company.

There are three sections of the project:

1. Most of the time was spent in this first section. I used unsupervised learning techniques to describe the relationship between the demographics of the company’s existing customers and the general population of Germany. This helped to describe parts of the general population that are more likely to be part of the mail-order company’s main customer base, and which parts of the general population are less so.

2. I built a supervised prediction model that is able to decide whether or not it will be worth to include an individual in the campaign. 

3. In the last section, I created predictions on the unseen mailout test partition, where the “RESPONSE” column had been withheld. I submitted the predictions on the test data in a form of CSV file as part of the Kaggle competition

This Capstone covers:

* Customer Segmentation * Kmeans Clustering * Python Programming * Supervised and Unsupervised Learning * Logistic Regression * Principal Component Analysis * Ensemble Learning * Random Forest * Gradient Boosting *Ada Boosting
* Random Sampling * SMOTE * MSMOTE * Extreme Gradient Boosting * Random Search Optimization * Hyperopt Automated Tuning
* Support Vector Machines * Kaggle Competition *

## Installation
This capstone project requires Python 3.x and the following Python libraries installed:

* [Numpy](https://numpy.org/) 
* [Matplotlib](https://matplotlib.org/ )
* [Pandas]( https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Hyperopt](https://pypi.org/project/hyperopt/)
* [Anaconda Individual Edition](https://www.anaconda.com/products/individual)

## Data 
There are four data files associated with this project:
• Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
• Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
• Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
•  Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

## Acknowledgments

I would like to thank [Arvato-Bertelsmann](https://www.bertelsmann.com/divisions/arvato/#st-1) for providing the data and [Udacity](https://www.udacity.com/) for a challening real world project.
