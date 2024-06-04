# Classification-Algorithms-Comparison
**Work in progress**
This projects aims to make a comparison of accuracy and confusion matrixes of the most popular classification algorithms. The classifications are made on Breast Cancer Wisconsin Dataset dwonloaded from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
and also available under this url: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## K-nearest neighbours:
KNN is a supervised learning classification algorithm that makes predictions based on the distances of the K nearest observations, as its name suggests. It is one of the most popular and simple classification algorithms in today's machine learning. 

The mean cross-validation score of this model is 0.96, which means that on average, the model correctly predicts the target variable 96% of the time.
The confusion matrix of those cross-validated outcomes presents as follows:

 ![KNN-cross-validation-confusion-matrix](https://github.com/SzymonScib/Classification-Algorithms-Comparison/assets/147078927/69304310-6f12-4068-ad62-5fd1589437d4)

After inspecting the confusion matrix we can calculate that this model predicts malignant tumors with ~93,3% accuracy, and benign tumors with ~98,3% accuracy.

## Logistic regression
Logistic regression is a supervised machine learning algorithm that estimates the probability of a binary event occurring, based on a given data set of independent variables.

The mean cross-validation score of this model is 0.98, which means that on average, the model correctly predicts the target variable 98% of the time.
The confusion matrix of those cross-validated outcomes presents as follows:

![Logistic_regression_confusion_matrix — kopia](https://github.com/SzymonScib/Classification-Algorithms-Comparison/assets/147078927/d0916139-a61b-4689-8a8f-0970a929d499)

After inspecting the confusion matrix we can calculate that this model predicts malignant tumors with ~96,2% accuracy, and benign tumors with ~99,2% accuracy. 

## Random forest
A random forest (RF) is an ensemble of decision trees in which each decision tree is trained with a specific random noise. Random forests are the most popular form of decision tree ensemble.

The mean cross-validation score of this model is 0.96, which means that on average, the model correctly predicts the target variable 96% of the time.
The confusion matrix of those cross-validated outcomes presents as follows:

![Random_forest_confusion_matrix](https://github.com/SzymonScib/Classification-Algorithms-Comparison/assets/147078927/0ea17bd6-7c36-413b-8b40-ab82d7908f66)

After inspecting the confusion matrix we can calculate that this model predicts malignant tumors with ~93,9% accuracy, and benign tumors with ~97,7% accuracy.

 
