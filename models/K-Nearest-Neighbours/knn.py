from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from math import sqrt
import pandas as pd

# Load the dataset
df = pd.read_csv('./data/cancer_data.csv')

# Splitting the dataset into features "X" - the measurements used to determine whether a tumor is malignant/benign, and labels "Y" - the diagnosis 
X = df.drop(['diagnosis', 'id'], axis=1)
Y = df['diagnosis']

# Use the Square Root of N rule to pick an optimal k
k = int(sqrt(len(X)))

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsClassifier(n_neighbors=k))  
])

cv_predictions = cross_val_predict(pipeline, X, Y, cv=5)

conf_matrix = confusion_matrix(Y, cv_predictions, labels=['M', 'B'])
print('Confusion Matrix for all cross-validated predictions:')
print(conf_matrix)

class_report = classification_report(Y, cv_predictions, target_names=['M', 'B'], labels=['M', 'B'])
print('Classification Report for all cross-validated predictions:')
print(class_report)

cv_scores = cross_val_score(pipeline, X, Y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')