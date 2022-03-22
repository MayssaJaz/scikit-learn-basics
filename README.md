# scikit-learn-basics

## General idea on this project
This project was made to answer this lab's questions: **[TP3](resources/TP3-DM.pdf)** .It aims to:

- Getting started with Python's scikit-learn library, dedicated to machine learning.

- Becoming familiar with the evaluation of models learned in supervised classification.
## Small walkthrough
These are the steps to follow in order to perform tests and choose the appropriate classifier in supervised learning
- Splitting our dataset into two parts: 2/3 for training and 1/3 for testing (in general)

<img src="https://github.com/MayssaJaz/scikit-learn-basics/blob/main/resources/training_test.jpeg" />

- Training our model on the training set which provides a prediction on the test set
- Testing results on the test set (calculating error/accuracy...)

```
Cross validation may be applied to ensure more accuracy to our prediction results
```
<img src="https://github.com/MayssaJaz/scikit-learn-basics/blob/main/resources/crossvalidation.png" />


## Requirements
- Python3 installed

- The different required dependencies are found in this file [requirements.txt](requirements/requirements.txt).Run this command to install them:

```
pip install -r requirements.txt
```
