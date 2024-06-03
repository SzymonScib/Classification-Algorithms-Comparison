# Classification-Algorithms-Comparison
**Work in progress**
This projects aims to make a comparison of accuracy and confusion matrixes of the most popular classification algorithms. The classifications are made on Breast Cancer Wisconsin Dataset dwonloaded from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
and also available under this url: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## K-nearest neighbours:
KNN is a supervised learning classification algorithm that makes predictions based on the distances of the K nearest observations, as its name suggests. It is one of the most popular and simple classification algorithms in today's machine learning. 

The mean cross-validation score of this model is 0.95, which means that on average, the model correctly predicts the target variable 95% of the time.
The confusion matrix of those cross-validated outcomes presents as follows:
 ![Knn_matrix](https://github.com/SzymonScib/Classification-Algorithms-Comparison/assets/147078927/8eb68dfa-64da-469c-bad2-639fe2defbfa)
 
We can observe that the model produces more false negatives (FN) than false positives (FP). This suggests that out of the 212 patients with malignant tumors in the test data, the model correctly classified 185 of them while misclassifying 27. This results in an approximate accuracy of ~88.2% in diagnosing malignant tumors based on the provided dataset.
